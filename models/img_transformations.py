from abc import ABC, abstractmethod
import torch
import torchaudio.transforms as T
from utils.utils_data import MinMaxScaler, MinMaxArgs


class TsImgEmbedder(ABC):
    """
    Abstract class for transforming time series to images and vice versa
    """

    def __init__(self, device, seq_len):
        self.device = device
        self.seq_len = seq_len

    @abstractmethod
    def ts_to_img(self, signal):
        """

        Args:
            signal: given time series

        Returns:
            image representation of the signal

        """
        pass

    @abstractmethod
    def img_to_ts(self, img):
        """

        Args:
            img: given generated image

        Returns:
            time series representation of the generated image
        """
        pass


class STFTEmbedder(TsImgEmbedder):
    """
    STFT transformation
    """

    def __init__(self, device, seq_len, n_fft, hop_length):
        super().__init__(device, seq_len)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_real, self.max_real, self.min_imag, self.max_imag = None, None, None, None

    def cache_min_max_params(self, train_data):
        """
        Args:
            train_data: training timeseries dataset. shape: B*L*K
        this function initializes the min and max values for the real and imaginary parts.
        we'll use this function only once, before the training loop starts.
        """
        real, imag = self.stft_transform(train_data)
        # compute and cache min and max values
        real, min_real, max_real = MinMaxScaler(real.numpy(), True)
        imag, min_imag, max_imag = MinMaxScaler(imag.numpy(), True)
        self.min_real, self.max_real = torch.Tensor(min_real), torch.Tensor(max_real)
        self.min_imag, self.max_imag = torch.Tensor(min_imag), torch.Tensor(max_imag)

    def stft_transform(self, data):
        """
        Args:
            data: time series data. Shape: B*L*K
        Returns:
            real and imaginary parts of the STFT transformation
        """
        data = torch.permute(data, (0, 2, 1))  # we permute to match requirements of torchaudio.transforms.Spectrogram
        # print("test",data.shape)
        spec = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, center=True, power=None).to(data.device)
        transformed_data = spec(data)
        return transformed_data.real, transformed_data.imag

    def ts_to_img(self, signal):
        assert self.min_real is not None, "use init_norm_args() to compute scaling arguments"
        # convert to complex spectrogram
        real, imag = self.stft_transform(signal)
        # MinMax scaling
        real = (MinMaxArgs(real, self.min_real.to(self.device), self.max_real.to(self.device)) - 0.5) * 2
        imag = (MinMaxArgs(imag, self.min_imag.to(self.device), self.max_imag.to(self.device)) - 0.5) * 2
        # stack real and imag parts
        stft_out = torch.cat((real, imag), dim=1)
        return stft_out

    def img_to_ts(self, x_image):
        n_fft = self.n_fft
        hop_length, length = self.hop_length, self.seq_len
        min_real, max_real, min_imag, max_imag = self.min_real.to(
            self.device), self.max_real.to(
            self.device), \
            self.min_imag.to(self.device), self.max_imag.to(
            self.device)
        # -- combine real and imaginary parts --
        split = torch.split(x_image, x_image.shape[1] // 2,
                            dim=1)  # x_image.shape[1] is twice the size of the original dim

        real, imag = split[0], split[1]
        unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
        unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
        unnormalized_stft = torch.complex(unnormalized_real, unnormalized_imag)
        # -- inverse stft --
        ispec = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length, center=True).to(self.device)

        x_time_series = ispec(unnormalized_stft, length)

        return torch.permute(x_time_series, (0, 2, 1))  # B*L*K(C)
