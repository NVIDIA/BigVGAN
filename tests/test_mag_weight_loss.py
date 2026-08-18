# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Regression test for the mag_weight term of MultiScaleMelSpectrogramLoss.
#
# Bug: the magnitude (mag_weight) portion of the loss was computed between
# the *log*-mel spectrograms instead of the *raw* mel spectrograms, so
# mag_weight > 0 added a second copy of the log-mel loss rather than a
# magnitude-domain loss. Latent with the default mag_weight=0.0.
#
# CPU-only; does not require the fused CUDA kernels.
# Run with:
#   uv run --with pytest --with torch --with librosa --with scipy \
#       pytest tests/test_mag_weight_loss.py

import os
import sys

# to import modules from parent_dir
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import torch

from loss import MultiScaleMelSpectrogramLoss

SR = 16000
N_MELS = 16
WINDOW_LENGTH = 256
CLAMP_EPS = 1e-5

MEL_KWARGS = dict(
    n_mels=N_MELS,
    fmin=0.0,
    fmax=None,
    window_length=WINDOW_LENGTH,
    hop_length=WINDOW_LENGTH // 4,
    match_stride=False,
    window_type="hann",
)


def _make_loss(mag_weight, log_weight):
    return MultiScaleMelSpectrogramLoss(
        sampling_rate=SR,
        n_mels=[N_MELS],
        window_lengths=[WINDOW_LENGTH],
        mel_fmin=[0.0],
        mel_fmax=[None],
        mag_weight=mag_weight,
        log_weight=log_weight,
    )


def _signals():
    torch.manual_seed(0)
    x = torch.randn(1, 1, 4096)
    y = torch.randn(1, 1, 4096)
    return x, y


def _logmel(mels):
    return torch.log(mels.clamp(min=CLAMP_EPS)) / torch.log(torch.tensor(10.0))


def test_mag_weight_uses_raw_mel_magnitude():
    """With log_weight=0 and mag_weight=1, the loss must equal the L1
    distance between the raw mel spectrograms (not the log-mels)."""
    x, y = _signals()
    loss_fn = _make_loss(mag_weight=1.0, log_weight=0.0)

    x_mels = loss_fn.mel_spectrogram(x, **MEL_KWARGS)
    y_mels = loss_fn.mel_spectrogram(y, **MEL_KWARGS)
    expected = torch.nn.functional.l1_loss(x_mels, y_mels)
    wrong = torch.nn.functional.l1_loss(_logmel(x_mels), _logmel(y_mels))

    # sanity: the two objectives differ, so the assertion below can tell
    # them apart
    assert not torch.isclose(expected, wrong)

    actual = loss_fn(x, y)
    assert torch.isclose(actual, expected, atol=1e-6), (
        f"mag_weight term must use raw mels: got {actual.item():.6f}, "
        f"expected raw-mel L1 {expected.item():.6f} "
        f"(log-mel L1 would be {wrong.item():.6f})"
    )


def test_default_mag_weight_zero_is_log_mel_only():
    """Default path (mag_weight=0.0) must equal the log-mel L1 loss and be
    unaffected by the fix."""
    x, y = _signals()
    loss_fn = _make_loss(mag_weight=0.0, log_weight=1.0)

    x_logmels = _logmel(loss_fn.mel_spectrogram(x, **MEL_KWARGS))
    y_logmels = _logmel(loss_fn.mel_spectrogram(y, **MEL_KWARGS))
    expected = torch.nn.functional.l1_loss(x_logmels, y_logmels)

    actual = loss_fn(x, y)
    assert torch.isclose(actual, expected, atol=1e-6)
