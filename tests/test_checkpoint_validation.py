import pytest
import torch
from utils import load_checkpoint


def test_load_checkpoint_raises_for_missing_file():
    missing_path = "/nonexistent/path/checkpoint.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(missing_path, device="cpu")
