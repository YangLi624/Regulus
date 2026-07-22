"""Automatic mixed-precision compatibility tests."""

import torch

from regulus.utils.amp import autocast_context, make_grad_scaler


def test_disabled_amp_helpers_on_cpu():
    scaler = make_grad_scaler(enabled=False)
    assert not scaler.is_enabled()
    value = torch.tensor([1.0])
    with autocast_context("cpu", enabled=False):
        output = value + 1
    assert output.item() == 2.0
