"""Regulus heterogeneous graph construction, pretraining, and frozen loading."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "HeteroDataBuilder",
    "FrozenHGTBundle",
    "HeteroRegulatorNet",
    "HeteroLoss",
    "create_loss_function",
    "Trainer",
    "run_graph_training",
]

_EXPORTS = {
    "HeteroDataBuilder": ("regulus.graph.build", "HeteroDataBuilder"),
    "FrozenHGTBundle": ("regulus.graph.frozen_hgt", "FrozenHGTBundle"),
    "HeteroRegulatorNet": ("regulus.graph.model", "HeteroRegulatorNet"),
    "HeteroLoss": ("regulus.graph.losses", "HeteroLoss"),
    "create_loss_function": ("regulus.graph.losses", "create_loss_function"),
    "Trainer": ("regulus.graph.hgt_trainer", "Trainer"),
    "run_graph_training": ("regulus.graph.train", "run_graph_training"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
