import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import tensorflow as tf

PROJECT_ROOT = Path("<your_path>/T2I-Diff")
DATA_DIR = PROJECT_ROOT / "logs" / "MDD" / "generated_data"

DATASET = "MDD"
FOLD = 1
SEED = 0
ITERATIONS = 5
USE_SUBSET_FOR_CORR = True

ori_full_path = DATA_DIR / f"ori_sig_{DATASET}_fold{FOLD}.npy"
gen_full_path = DATA_DIR / f"gen_sig_{DATASET}_fold{FOLD}.npy"

sys.path.append(str(PROJECT_ROOT))

from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics

np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_generated_data(ori_path, gen_path):
    ori_data = np.load(ori_path)
    fake_data = np.load(gen_path)

    print("Original data shape:", ori_data.shape)
    print("Generated data shape:", fake_data.shape)

    return ori_data, fake_data


def minmax_normalize_roiwise(ori_data, fake_data):
    """
    ROI-wise min-max normalization for fMRI signals.

    Shape:
        ori_data:  (subjects, timesteps, ROI)
        fake_data: (subjects, timesteps, ROI)

    Min/max are computed from real data across subjects and timesteps
    for each ROI, then reused for both real and generated data.
    """
    data_min = ori_data.min(axis=(0, 1), keepdims=True)
    data_max = ori_data.max(axis=(0, 1), keepdims=True)

    ori_norm = (ori_data - data_min) / (data_max - data_min + 1e-8)
    fake_norm = (fake_data - data_min) / (data_max - data_min + 1e-8)

    return ori_norm, fake_norm


def sample_indices(num_samples, num_select):
    """
    Randomly select samples without replacement.
    """
    num_select = min(num_select, num_samples)
    return np.random.choice(num_samples, size=num_select, replace=False)


def compute_correlational_score(ori_data, fake_data, iterations=5, use_subset=True):
    print("\nCompute correlational score")

    x_real = torch.from_numpy(ori_data).float()
    x_fake = torch.from_numpy(fake_data).float()

    scores = []

    if use_subset:
        num_select = max(1, x_real.shape[0] // iterations)
    else:
        num_select = x_real.shape[0]

    for i in range(iterations):
        if use_subset:
            real_idx = sample_indices(x_real.shape[0], num_select)
            fake_idx = sample_indices(x_fake.shape[0], num_select)

            real_batch = x_real[real_idx]
            fake_batch = x_fake[fake_idx]
        else:
            real_batch = x_real
            fake_batch = x_fake[:x_real.shape[0]]

        corr = CrossCorrelLoss(real_batch, name="CrossCorrelLoss")
        loss = corr.compute(fake_batch)

        scores.append(loss.item())
        print(f"Iter {i}: cross-correlation = {loss.item()}")

    display_scores(scores)
    return scores


def compute_context_fid_score(ori_data, fake_data, iterations=5):
    context_fid_score = []
    print("\nCompute Context-FID score")

    for i in range(iterations):
        context_fid = Context_FID(ori_data[:], fake_data[:ori_data.shape[0]])
        context_fid_score.append(context_fid)
        print(f'Iter {i}: ', 'context-fid =', context_fid, '\n')


    display_scores(context_fid_score)
    return context_fid_score


def compute_discriminative_score(ori_data, fake_data, iterations=5):
    print("\nCompute discriminative score")

    scores = []
    fake_data = fake_data[:ori_data.shape[0]]

    for i in range(iterations):
        temp_disc, fake_acc, real_acc = discriminative_score_metrics(
            ori_data,
            fake_data
        )

        scores.append(temp_disc)

        print(
            f"Iter {i}: "
            f"discriminative score = {temp_disc}, "
            f"fake acc = {fake_acc}, "
            f"real acc = {real_acc}"
        )

    display_scores(scores)
    return scores


def compute_predictive_score(ori_data, fake_data, iterations=5):
    print("\nCompute predictive score")

    scores = []
    fake_data = fake_data[:ori_data.shape[0]]

    for i in range(iterations):
        temp_pred = predictive_score_metrics(
            ori_data,
            fake_data
        )

        scores.append(temp_pred)
        print(f"Iter {i}: predictive score = {temp_pred}")

    display_scores(scores)
    return scores


ori_data, fake_data = load_generated_data(
    ori_full_path,
    gen_full_path
)

ori_data, fake_data = minmax_normalize_roiwise(
    ori_data,
    fake_data
)

correlational_scores = compute_correlational_score(
    ori_data,
    fake_data,
    iterations=ITERATIONS,
    use_subset=USE_SUBSET_FOR_CORR
)

context_fid_score = compute_context_fid_score(
    ori_data,
    fake_data,
    iterations=ITERATIONS
)

discriminative_scores = compute_discriminative_score(
    ori_data,
    fake_data,
    iterations=ITERATIONS
)

predictive_scores = compute_predictive_score(
    ori_data,
    fake_data,
    iterations=ITERATIONS
)