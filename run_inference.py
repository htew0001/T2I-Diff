import os
import sys
import logging

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import torch.multiprocessing

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset

from metrics import evaluate_model_uncond
from utils.loggers import PrintLogger
from models.model import T2iDiff
from models.sampler import DiffusionProcess
from utils.utils import (
    save_checkpoint,
    restore_state,
    create_model_name_and_dir,
    print_model_params,
    log_config_and_tags,
)
from utils.utils_args import parse_args_cond_fmri

N_FOLDS = 5
SEED = 42
# OUTER_FOLD = 1
# INNER_FOLD = 1

# If True, print all fold shapes and class distributions
PRINT_ALL_FOLDS = True

def save_generated_data(args, ori_sig, gen_sig, sampling_labels):
    save_dir = args.gen_folderpath
    os.makedirs(save_dir, exist_ok=True)

    gen_path = os.path.join(save_dir, f"gen_sig_{args.dataset}_fold{args.outfold_num}.npy")
    ori_path = os.path.join(save_dir, f"ori_sig_{args.dataset}_fold{args.outfold_num}.npy")
    label_path = os.path.join(save_dir, f"label_sig_{args.dataset}_fold{args.outfold_num}.npy")

    np.save(ori_path, ori_sig)
    np.save(gen_path, gen_sig)
    np.save(label_path, sampling_labels)

    print(f"Saved generated data to: {save_dir}")
    
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_sampling_labels_from_config(args):
    sampling_labels = []

    print(f"synthetic samples distribution:")
    for cls, count in args.sampling["class_counts"].items():
        sampling_labels.extend([cls] * count)
        print(f" Class {cls}: {count}")

    return np.array(sampling_labels)


def print_label_distribution(labels, name="Labels"):
    unique_classes, counts = np.unique(labels, return_counts=True)
    print(f"\n{name} distribution:")
    for cls, count in zip(unique_classes, counts):
        print(f"  Class {cls}: {count}")


def print_data_statistics(data):
    print("\nData Statistics:")
    print(f"  Min value : {np.min(data)}")
    print(f"  Max value : {np.max(data)}")
    print(f"  Mean value: {np.mean(data)}")
    print(f"  Std dev   : {np.std(data)}")


def build_nested_folds(labels, n_folds=5, seed=42):
    fold_idx = {}

    dummy_data = np.zeros((len(labels), 1))
    skf_outer = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for out_c, (outer_train_idx, outer_test_idx) in enumerate(skf_outer.split(dummy_data, labels), 1):
        inner_data = dummy_data[outer_train_idx]
        inner_labels = labels[outer_train_idx]

        skf_inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        for in_c, (train_idx2, val_idx1) in enumerate(skf_inner.split(inner_data, inner_labels), 1):
            inner_train_idx = outer_train_idx[train_idx2]
            inner_val_idx = outer_train_idx[val_idx1]

            np.random.shuffle(inner_train_idx)

            fold_key = f"outer{out_c}_inner{in_c}"
            fold_idx[fold_key] = {
                "outer_train": outer_train_idx,
                "outer_test": outer_test_idx,
                "inner_train": inner_train_idx,
                "inner_val": inner_val_idx,
            }

    return fold_idx


def get_fold_data(data, labels, fold_idx, split, o_fold, i_fold):
    key = f"outer{o_fold}_inner{i_fold}"

    if split == "train":
        indices = fold_idx[key]["inner_train"]
    elif split == "val":
        indices = fold_idx[key]["inner_val"]
    elif split == "test":
        indices = fold_idx[key]["outer_test"]
    elif split == "outer_train":
        indices = fold_idx[key]["outer_train"]
    else:
        raise ValueError(f"Unknown split: {split}")

    return data[indices], labels[indices]


def print_fold_summary(ts_data, labels, fold_idx, n_folds=5):
    for o_fold in range(1, n_folds + 1):
        for i_fold in range(1, n_folds + 1):
            X_train, y_train = get_fold_data(ts_data, labels, fold_idx, "train", o_fold, i_fold)
            X_val, y_val = get_fold_data(ts_data, labels, fold_idx, "val", o_fold, i_fold)
            X_test, y_test = get_fold_data(ts_data, labels, fold_idx, "test", o_fold, i_fold)
            X_outer_train, y_outer_train = get_fold_data(ts_data, labels, fold_idx, "outer_train", o_fold, i_fold)

            print(f"\nOUTER {o_fold}, INNER {i_fold}")
            print("  TRAIN      :", X_train.shape, dict(zip(*np.unique(y_train, return_counts=True))))
            print("  VAL        :", X_val.shape, dict(zip(*np.unique(y_val, return_counts=True))))
            print("  OUTER_TRAIN:", X_outer_train.shape, dict(zip(*np.unique(y_outer_train, return_counts=True))))
            print("  TEST       :", X_test.shape, dict(zip(*np.unique(y_test, return_counts=True))))
            print("-" * 50)


class MyDataset(Dataset):
    def __init__(self, X, Y=None, target_len=256):
        self.x = torch.tensor(X, dtype=torch.float32)
        self.y = None if Y is None else torch.tensor(Y, dtype=torch.long)
        self.target_len = target_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x
        x = self.pad(x)

        if self.y is None:
            return x

        return x[idx], self.y[idx]

    def pad(self, x):
        L = x.shape[1]

        if L < self.target_len:
            pad_len = self.target_len - L
            x = F.pad(x,(0,0,0,pad_len))

        return x


def main():
    set_seed(SEED)
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # sys.argv = ["run_inference.py", "--config", CONFIG_PATH]
    args = parse_args_cond_fmri()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nUsing device:", args.device)

    mat = sio.loadmat(args.datapath)
    ts_data = mat["fmridata"].transpose(2, 1, 0).astype(np.float32)
    labels = mat["labels"].squeeze().astype(int)

    # output shape: (num_subjects, time_length, num_ROIs)
    print("ts_data shape:", ts_data.shape)
    print("labels shape:", labels.shape)

    print_label_distribution(labels, "Full dataset")
    print_data_statistics(ts_data)

    fold_idx = build_nested_folds(labels, n_folds=N_FOLDS, seed=SEED)

    if PRINT_ALL_FOLDS:
        print_fold_summary(ts_data, labels, fold_idx, n_folds=N_FOLDS)

    X_outer_train, y_outer_train = get_fold_data(
        ts_data, labels, fold_idx, "outer_train", args.outfold_num, args.infold_num
    )
    X_train, y_train = get_fold_data(ts_data, labels, fold_idx, "train", args.outfold_num, args.infold_num)
    X_val, y_val = get_fold_data(ts_data, labels, fold_idx, "val", args.outfold_num, args.infold_num)
    X_test, y_test = get_fold_data(ts_data, labels, fold_idx, "test", args.outfold_num, args.infold_num)

    print("\nSelected fold:")
    print("  X_outer_train:", X_outer_train.shape)
    print("  X_train      :", X_train.shape)
    print("  X_val        :", X_val.shape)
    print("  X_test       :", X_test.shape)

    name = create_model_name_and_dir(args)
    logging.info(args)

    logger = PrintLogger()
    log_config_and_tags(args, logger, name)

    target_len = X_outer_train.shape[1]

    sampling_labels = create_sampling_labels_from_config(args)
    train_dataset = MyDataset(X_outer_train, sampling_labels, target_len=target_len)
   
    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    logging.info(args.dataset + " dataset is ready.")

    model = T2iDiff(args=args, device=args.device).to(args.device)

    if args.use_stft:
        model.init_stft_embedder(train_loader)

    state = dict(model=model, epoch=0)
    ema_model = model.model_ema if args.ema else None
    restore_state(args, state, ema_model=ema_model)

    print_model_params(logger, model)

    logging.info(f"Start sampling.")

    gen_sig = []
    real_sig = []
    model.eval()
    with torch.no_grad():
        with model.ema_scope():
            process = DiffusionProcess(args,model.net,
                        (args.input_channels, args.img_resolution, 24),
                    )
            for batch in tqdm(train_loader, desc="Sampling"):
                x_ts_r = batch[0].to(args.device)
                y = batch[1].to(args.device)
                y_onehot = F.one_hot(y, num_classes=args.num_classes).float()

                x_img_sampled = process.sampling(
                        sampling_number=x_ts_r.shape[0],
                        labels=y_onehot,
                )

                x_ts = model.img_to_ts(x_img_sampled)

                gen_sig.append(x_ts.detach().cpu().numpy())
                real_sig.append(batch[0].detach().cpu().numpy())

    gen_sig = np.vstack(gen_sig)
    real_sig = np.vstack(real_sig)

    logging.info("Data generation is complete")
    print("gen :", gen_sig.shape)
    print("real:", real_sig.shape)

    logging.info("Start saving data")
    save_generated_data(args, real_sig, gen_sig, sampling_labels)

if __name__ == "__main__":
    main()
