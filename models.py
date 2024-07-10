# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.


import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torchaudio.transforms import Spectrogram, Resample
from librosa.filters import mel as librosa_mel_fn
from scipy import signal

import activations
from utils import init_weights, get_padding
from alias_free_torch.act import Activation1d as TorchActivation1d
import typing
from typing import List, Optional, Tuple
from collections import namedtuple
import math
import functools


class AMPBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5), activation=None):
        super(AMPBlock1, self).__init__()
        self.h = h

        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(self.convs2) # total number of conv layers

        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class AMPBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3), activation=None):
        super(AMPBlock2, self).__init__()
        self.h = h

        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

        self.num_layers = len(self.convs) # total number of conv layers

        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        if activation == 'snake': # periodic nonlinearity with snake function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.Snake(channels, alpha_logscale=h.snake_logscale))
                for _ in range(self.num_layers)
            ])
        elif activation == 'snakebeta': # periodic nonlinearity with snakebeta function and anti-aliasing
            self.activations = nn.ModuleList([
                Activation1d(
                    activation=activations.SnakeBeta(channels, alpha_logscale=h.snake_logscale))
                 for _ in range(self.num_layers)
            ])
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")

    def forward(self, x):
        for c, a in zip (self.convs, self.activations):
            xt = a(x)
            xt = c(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class BigVGAN(torch.nn.Module):
    # this is our main BigVGAN model. Applies anti-aliased periodic activation for resblocks.
    # New in v2: if use_cuda_kernel is set to True, it loads optimized CUDA kernels for AMP.
    # NOTE: use_cuda_kernel=True should be used for inference only (training is not supported).
    def __init__(
        self,
        h,
        use_cuda_kernel: bool=False
    ):
        super(BigVGAN, self).__init__()
        self.h = h
        self.h["use_cuda_kernel"] = use_cuda_kernel # add it to global hyperparameters (h)

        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)

        # pre conv
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))

        # define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock = AMPBlock1 if h.resblock == '1' else AMPBlock2

        # transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(nn.ModuleList([
                weight_norm(ConvTranspose1d(h.upsample_initial_channel // (2 ** i),
                                            h.upsample_initial_channel // (2 ** (i + 1)),
                                            k, u, padding=(k - u) // 2))
            ]))

        # residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d, activation=h.activation))
        
        # select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if self.h.get("use_cuda_kernel", False):
            # faster CUDA kernel implementation of Activation1d
            from alias_free_cuda.activation1d import Activation1d as CudaActivation1d
            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # post conv
        if h.activation == "snake": # periodic nonlinearity with snake function and anti-aliasing
            activation_post = activations.Snake(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        elif h.activation == "snakebeta": # periodic nonlinearity with snakebeta function and anti-aliasing
            activation_post = activations.SnakeBeta(ch, alpha_logscale=h.snake_logscale)
            self.activation_post = Activation1d(activation=activation_post)
        else:
            raise NotImplementedError("activation incorrectly specified. check the config file and look for 'activation'.")
        
        # whether to use bias for the final conv_post. Defaults to True for backward compatibility
        self.use_bias_at_final = h.get("use_bias_at_final", True)
        self.conv_post = weight_norm(Conv1d(
            ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final
        ))

        # weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)
        
        # final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = h.get("use_tanh_at_final", True)

    def forward(self, x):
        # pre conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # post conv
        x = self.activation_post(x)
        x = self.conv_post(x)
        # final tanh activation
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1., max=1.) # bound the output to [-1, 1]

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            for l_i in l:
                remove_weight_norm(l_i)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, h, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = h.discriminator_channel_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = h.mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        discriminators = [DiscriminatorP(h, rs, use_spectral_norm=h.use_spectral_norm) for rs in self.mpd_reshapes]
        self.discriminators = nn.ModuleList(discriminators)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorR(nn.Module):
    def __init__(self, cfg, resolution):
        super().__init__()

        self.resolution = resolution
        assert len(self.resolution) == 3, \
            "MRD layer requires list with len=3, got {}".format(self.resolution)
        self.lrelu_slope = 0.1

        norm_f = weight_norm if cfg.use_spectral_norm == False else spectral_norm
        if hasattr(cfg, "mrd_use_spectral_norm"):
            print("INFO: overriding MRD use_spectral_norm as {}".format(cfg.mrd_use_spectral_norm))
            norm_f = weight_norm if cfg.mrd_use_spectral_norm == False else spectral_norm
        self.d_mult = cfg.discriminator_channel_mult
        if hasattr(cfg, "mrd_channel_mult"):
            print("INFO: overriding mrd channel multiplier as {}".format(cfg.mrd_channel_mult))
            self.d_mult = cfg.mrd_channel_mult

        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(32*self.d_mult), (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(32 * self.d_mult), 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []

        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        x = torch.view_as_real(x)  # [B, F, TT, 2]
        mag = torch.norm(x, p=2, dim =-1) #[B, F, TT]

        return mag


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, cfg, debug=False):
        super().__init__()
        self.resolutions = cfg.resolutions
        assert len(self.resolutions) == 3,\
            "MRD requires list of list with len=3, each element having a list with len=3. got {}".\
                format(self.resolutions)
        self.discriminators = nn.ModuleList(
            [DiscriminatorR(cfg, resolution) for resolution in self.resolutions]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# Method based on descript-audio-codec: https://github.com/descriptinc/descript-audio-codec
# Modified code adapted from https://github.com/gemelo-ai/vocos under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorB(nn.Module):
    def __init__(
        self,
        window_length: int,
        channels: int = 32,
        hop_factor: float = 0.25,
        bands: Tuple[Tuple[float, float], ...] = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)),
    ):
        super().__init__()
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.spec_fn = Spectrogram(
            n_fft=window_length, hop_length=int(window_length * hop_factor), win_length=window_length, power=None
        )
        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands
        convs = lambda: nn.ModuleList(
            [
                weight_norm(nn.Conv2d(2, channels, (3, 9), (1, 1), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 9), (1, 2), padding=(1, 4))),
                weight_norm(nn.Conv2d(channels, channels, (3, 3), (1, 1), padding=(1, 1))),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])

        self.conv_post = weight_norm(nn.Conv2d(channels, 1, (3, 3), (1, 1), padding=(1, 1)))

    def spectrogram(self, x):
        # Remove DC offset
        x = x - x.mean(dim=-1, keepdims=True)
        # Peak normalize the volume of input audio
        x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
        x = self.spec_fn(x)
        x = torch.view_as_real(x)
        x = x.permute(0, 3, 2, 1) # [B, F, T, C] -> [B, C, T, F]
        # Split into bands
        x_bands = [x[..., b[0] : b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor):
        x_bands = self.spectrogram(x.squeeze(1))
        fmap = []
        x = []
        
        for band, stack in zip(x_bands, self.band_convs):
            for i, layer in enumerate(stack):
                band = layer(band)
                band = torch.nn.functional.leaky_relu(band, 0.1)
                if i > 0:
                    fmap.append(band)
            x.append(band)
            
        x = torch.cat(x, dim=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return x, fmap

# Method based on descript-audio-codec: https://github.com/descriptinc/descript-audio-codec
# Modified code adapted from https://github.com/gemelo-ai/vocos under the MIT license.
#   LICENSE is in incl_licenses directory.
class MultiBandDiscriminator(nn.Module):
    def __init__(
        self,
        h,
    ):
        """
        Multi-band multi-scale STFT discriminator, with the architecture based on https://github.com/descriptinc/descript-audio-codec.
        and the modified code adapted from https://github.com/gemelo-ai/vocos.
        """
        super().__init__()
        # fft_sizes (list[int]): Tuple of window lengths for FFT. Defaults to [2048, 1024, 512] if not set in h.
        self.fft_sizes = h.get("mbd_fft_sizes", [2048, 1024, 512])
        self.discriminators = nn.ModuleList(
            [DiscriminatorB(window_length=w) for w in self.fft_sizes]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(x=y)
            y_d_g, fmap_g = d(x=y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Adapted from https://github.com/open-mmlab/Amphion/blob/main/models/vocoders/gan/discriminator/mssbcqtd.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class DiscriminatorCQT(nn.Module):
    def __init__(self, cfg, hop_length, n_octaves, bins_per_octave):
        super().__init__()
        self.cfg = cfg
        
        self.filters = cfg["cqtd_filters"]
        self.max_filters = cfg["cqtd_max_filters"]
        self.filters_scale = cfg["cqtd_filters_scale"]
        self.kernel_size = (3, 9)
        self.dilations = cfg["cqtd_dilations"]
        self.stride = (1, 2)

        self.in_channels = cfg["cqtd_in_channels"]
        self.out_channels = cfg["cqtd_out_channels"]
        self.fs = cfg["sampling_rate"]
        self.hop_length = hop_length
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave

        # lazy-load
        from nnAudio import features
        self.cqt_transform = features.cqt.CQT2010v2(
            sr=self.fs * 2,
            hop_length=self.hop_length,
            n_bins=self.bins_per_octave * self.n_octaves,
            bins_per_octave=self.bins_per_octave,
            output_format="Complex",
            pad_mode="constant",
        )

        self.conv_pres = nn.ModuleList()
        for i in range(self.n_octaves):
            self.conv_pres.append(
                nn.Conv2d(
                    self.in_channels * 2,
                    self.in_channels * 2,
                    kernel_size=self.kernel_size,
                    padding=self.get_2d_padding(self.kernel_size),
                )
            )

        self.convs = nn.ModuleList()

        self.convs.append(
            nn.Conv2d(
                self.in_channels * 2,
                self.filters,
                kernel_size=self.kernel_size,
                padding=self.get_2d_padding(self.kernel_size),
            )
        )

        in_chs = min(self.filters_scale * self.filters, self.max_filters)
        for i, dilation in enumerate(self.dilations):
            out_chs = min(
                (self.filters_scale ** (i + 1)) * self.filters, self.max_filters
            )
            self.convs.append(
                weight_norm(nn.Conv2d(
                    in_chs,
                    out_chs,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    dilation=(dilation, 1),
                    padding=self.get_2d_padding(self.kernel_size, (dilation, 1)),
                ))
            )
            in_chs = out_chs
        out_chs = min(
            (self.filters_scale ** (len(self.dilations) + 1)) * self.filters,
            self.max_filters,
        )
        self.convs.append(
            weight_norm(nn.Conv2d(
                in_chs,
                out_chs,
                kernel_size=(self.kernel_size[0], self.kernel_size[0]),
                padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
            ))
        )

        self.conv_post = weight_norm(nn.Conv2d(
            out_chs,
            self.out_channels,
            kernel_size=(self.kernel_size[0], self.kernel_size[0]),
            padding=self.get_2d_padding((self.kernel_size[0], self.kernel_size[0])),
        ))

        self.activation = torch.nn.LeakyReLU(negative_slope=0.1)
        self.resample = Resample(orig_freq=self.fs, new_freq=self.fs * 2)
        
        self.cqtd_normalize_volume = self.cfg.get("cqtd_normalize_volume", False)
        if self.cqtd_normalize_volume:
            print(f"INFO: cqtd_normalize_volume set to True. Will apply DC offset removal & peak volume normalization in CQTD!")
    
    def get_2d_padding(
            self, kernel_size: typing.Tuple[int, int], dilation: typing.Tuple[int, int] = (1, 1)
        ):
        return (
            ((kernel_size[0] - 1) * dilation[0]) // 2,
            ((kernel_size[1] - 1) * dilation[1]) // 2,
        )

    def forward(self, x):
        fmap = []
        
        if self.cqtd_normalize_volume:
            # Remove DC offset
            x = x - x.mean(dim=-1, keepdims=True)
            # Peak normalize the volume of input audio
            x = 0.8 * x / (x.abs().max(dim=-1, keepdim=True)[0] + 1e-9)
            
        x = self.resample(x)
        
        z = self.cqt_transform(x)

        z_amplitude = z[:, :, :, 0].unsqueeze(1)
        z_phase = z[:, :, :, 1].unsqueeze(1)

        z = torch.cat([z_amplitude, z_phase], dim=1)
        z = torch.permute(z, (0, 1, 3, 2)) # [B, C, W, T] -> [B, C, T, W]

        latent_z = []
        for i in range(self.n_octaves):
            latent_z.append(
                self.conv_pres[i](
                    z[
                        :,
                        :,
                        :,
                        i * self.bins_per_octave : (i + 1) * self.bins_per_octave,
                    ]
                )
            )
        latent_z = torch.cat(latent_z, dim=-1)

        for i, l in enumerate(self.convs):
            latent_z = l(latent_z)

            latent_z = self.activation(latent_z)
            fmap.append(latent_z)

        latent_z = self.conv_post(latent_z)

        return latent_z, fmap
    

class MultiScaleSubbandCQTDiscriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        # Using get with defaults
        self.cfg["cqtd_filters"] = self.cfg.get("cqtd_filters", 32)
        self.cfg["cqtd_max_filters"] = self.cfg.get("cqtd_max_filters", 1024)
        self.cfg["cqtd_filters_scale"] = self.cfg.get("cqtd_filters_scale", 1)
        self.cfg["cqtd_dilations"] = self.cfg.get("cqtd_dilations", [1, 2, 4])
        self.cfg["cqtd_in_channels"] = self.cfg.get("cqtd_in_channels", 1)
        self.cfg["cqtd_out_channels"] = self.cfg.get("cqtd_out_channels", 1)
        # multi-scale params to loop over
        self.cfg["cqtd_hop_lengths"] = self.cfg.get("cqtd_hop_lengths", [512, 256, 256])
        self.cfg["cqtd_n_octaves"] = self.cfg.get("cqtd_n_octaves", [9, 9, 9])
        self.cfg["cqtd_bins_per_octaves"] = self.cfg.get("cqtd_bins_per_octaves", [24, 36, 48])

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorCQT(
                    self.cfg,
                    hop_length=self.cfg["cqtd_hop_lengths"][i],
                    n_octaves=self.cfg["cqtd_n_octaves"][i],
                    bins_per_octave=self.cfg["cqtd_bins_per_octaves"][i],
                )
                for i in range(len(self.cfg["cqtd_hop_lengths"]))
            ]
        )

    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    

class CombinedDiscriminator(nn.Module):
    # wrapper of chaining multiple discrimiantor architectures
    # ex: combine mbd and cqtd as a single class
    def __init__(
        self,
        list_discriminator: List[nn.Module]
    ):
        super().__init__()
        self.discrimiantor = nn.ModuleList(list_discriminator)
        
    def forward(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        for disc in self.discrimiantor:
            y_d_r, y_d_g, fmap_r, fmap_g = disc(y, y_hat)
            y_d_rs.extend(y_d_r)
            fmap_rs.extend(fmap_r)
            y_d_gs.extend(y_d_g)
            fmap_gs.extend(fmap_g)
            
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


# Adapted from https://github.com/descriptinc/descript-audio-codec/blob/main/dac/nn/loss.py under the MIT license.
#   LICENSE is in incl_licenses directory.
class MultiScaleMelSpectrogramLoss(nn.Module):
    """Compute distance between mel spectrograms. Can be used
    in a multi-scale way.

    Parameters
    ----------
    n_mels : List[int]
        Number of mels per STFT, by default [5, 10, 20, 40, 80, 160, 320],
    window_lengths : List[int], optional
        Length of each window of each STFT, by default [32, 64, 128, 256, 512, 1024, 2048]
    loss_fn : typing.Callable, optional
        How to compare each loss, by default nn.L1Loss()
    clamp_eps : float, optional
        Clamp on the log magnitude, below, by default 1e-5
    mag_weight : float, optional
        Weight of raw magnitude portion of loss, by default 0.0 (no ampliciation on mag part)
    log_weight : float, optional
        Weight of log magnitude portion of loss, by default 1.0
    pow : float, optional
        Power to raise magnitude to before taking log, by default 1.0
    weight : float, optional
        Weight of this loss, by default 1.0
    match_stride : bool, optional
        Whether to match the stride of convolutional layers, by default False

    Implementation copied from: https://github.com/descriptinc/lyrebird-audiotools/blob/961786aa1a9d628cca0c0486e5885a457fe70c1a/audiotools/metrics/spectral.py
    Additional code copied and modified from https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
    """

    def __init__(
        self,
        sampling_rate: int,
        n_mels: List[int] = [5, 10, 20, 40, 80, 160, 320],
        window_lengths: List[int] = [32, 64, 128, 256, 512, 1024, 2048],
        loss_fn: typing.Callable = nn.L1Loss(),
        clamp_eps: float = 1e-5,
        mag_weight: float = 0.0,
        log_weight: float = 1.0,
        pow: float = 1.0,
        weight: float = 1.0,
        match_stride: bool = False,
        mel_fmin: List[float] = [0, 0, 0, 0, 0, 0, 0],
        mel_fmax: List[float] = [None, None, None, None, None, None, None],
        window_type: str = 'hann',
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        
        STFTParams = namedtuple(
            "STFTParams",
            ["window_length", "hop_length", "window_type", "match_stride"],
        )
        
        self.stft_params = [
            STFTParams(
                window_length=w,
                hop_length=w // 4,
                match_stride=match_stride,
                window_type=window_type,
            )
            for w in window_lengths
        ]
        self.n_mels = n_mels
        self.loss_fn = loss_fn
        self.clamp_eps = clamp_eps
        self.log_weight = log_weight
        self.mag_weight = mag_weight
        self.weight = weight
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.pow = pow
        
    @staticmethod
    @functools.lru_cache(None)
    def get_window(
        window_type,window_length, 
    ):
        return signal.get_window(window_type, window_length)
    
    @staticmethod
    @functools.lru_cache(None)
    def get_mel_filters(
        sr, n_fft, n_mels, fmin, fmax
    ):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        
    def mel_spectrogram(
        self, wav, n_mels, fmin, fmax, window_length, hop_length, match_stride, window_type
    ):
        # mirrors AudioSignal.mel_spectrogram used by BigVGAN-v2 training from:
        # https://github.com/descriptinc/audiotools/blob/master/audiotools/core/audio_signal.py
        B, C, T = wav.shape
        
        if match_stride:
            assert (
                hop_length == window_length // 4
            ), "For match_stride, hop must equal n_fft // 4"
            right_pad = math.ceil(T / hop_length) * hop_length - T
            pad = (window_length - hop_length) // 2
        else:
            right_pad = 0
            pad = 0
            
        wav = torch.nn.functional.pad(
            wav, (pad, pad + right_pad), mode='reflect'
        )
        
        window = self.get_window(window_type, window_length)
        window = torch.from_numpy(window).to(wav.device).float()
        
        stft = torch.stft(
            wav.reshape(-1, T),
            n_fft=window_length,
            hop_length=hop_length,
            window=window,
            return_complex=True,
            center=True,
        )
        _, nf, nt = stft.shape
        stft = stft.reshape(B, C, nf, nt)
        if match_stride:
            # Drop first two and last two frames, which are added
            # because of padding. Now num_frames * hop_length = num_samples.
            stft = stft[..., 2:-2]
        magnitude = torch.abs(stft)
        
        nf = magnitude.shape[2]
        mel_basis = self.get_mel_filters(self.sampling_rate, 2 * (nf - 1), n_mels, fmin, fmax)
        mel_basis = torch.from_numpy(mel_basis).to(wav.device)
        mel_spectrogram = magnitude.transpose(2, -1) @ mel_basis.T
        mel_spectrogram = mel_spectrogram.transpose(-1, 2)
        
        return mel_spectrogram
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Computes mel loss between an estimate and a reference
        signal.

        Parameters
        ----------
        x : torch.Tensor
            Estimate signal
        y : torch.Tensor
            Reference signal

        Returns
        -------
        torch.Tensor
            Mel loss.
        """
        
        loss = 0.0
        for n_mels, fmin, fmax, s in zip(
            self.n_mels, self.mel_fmin, self.mel_fmax, self.stft_params
        ):
            kwargs = {
                "n_mels": n_mels,
                "fmin": fmin,
                "fmax": fmax,
                "window_length": s.window_length,
                "hop_length": s.hop_length,
                "match_stride": s.match_stride,
                "window_type": s.window_type,
            }

            x_mels = self.mel_spectrogram(x, **kwargs)
            y_mels = self.mel_spectrogram(y, **kwargs)
            x_logmels = torch.log(x_mels.clamp(min=self.clamp_eps).pow(self.pow)) / torch.log(torch.tensor(10.0))
            y_logmels = torch.log(y_mels.clamp(min=self.clamp_eps).pow(self.pow)) / torch.log(torch.tensor(10.0))
            
            loss += self.log_weight * self.loss_fn(x_logmels, y_logmels)
            loss += self.mag_weight * self.loss_fn(x_logmels, y_logmels)
            
        return loss


# loss functions
def feature_loss(
    fmap_r: List[List[torch.Tensor]],
    fmap_g: List[List[torch.Tensor]]
) -> torch.Tensor:

    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2 # this equates to lambda=2.0 for the feature matching loss

def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_generated_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:

    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(
    disc_outputs: List[torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses