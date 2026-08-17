import torch.nn as nn
from utils import apply_weight_norm, init_weights


def test_init_weights_and_weight_norm_apply_only_to_conv_modules():
    conv = nn.Conv1d(1, 1, 3)
    linear = nn.Linear(3, 3)

    init_weights(conv, std=0.5)
    init_weights(linear, std=0.5)

    # Conv weights are re-initialized; linear weights stay at zero.
    assert conv.weight.std().item() > 0
    assert linear.weight.std().item() == 0

    apply_weight_norm(conv)
    apply_weight_norm(linear)

    # Weight normalization is applied only to Conv modules.
    assert hasattr(conv, "weight_g")
