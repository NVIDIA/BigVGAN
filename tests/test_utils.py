import numpy as np
import pytest
import torch
from unittest.mock import patch

from utils import MAX_WAV_VALUE, save_audio


def test_save_audio_clamps_before_int16_conversion():
    """Values outside [-1, 1] must be clipped before scaling to int16."""
    captured = {}

    def fake_write(path, sr, data):
        captured["data"] = data

    audio = torch.tensor([0.5, 1.5, -0.8, -2.0])
    expected = audio.clamp(min=-1.0, max=1.0).numpy() * MAX_WAV_VALUE

    with patch("utils.write", fake_write):
        save_audio(audio, "/tmp/fake.wav", 22050)

    np.testing.assert_array_equal(
        captured["data"],
        expected.astype("int16"),
