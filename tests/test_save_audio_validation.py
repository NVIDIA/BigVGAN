import os
import tempfile
import pytest
import torch
from utils import save_audio


def test_save_audio_rejects_non_1d_input():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "audio.wav")
        two_d = torch.zeros(2, 100)
        with pytest.raises(ValueError, match="expects 1D audio"):
            save_audio(two_d, path, sr=22050)
