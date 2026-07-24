"""Regulus perturbation driver inference and functional manipulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "1.0.0"
__all__ = ["RegulusModel", "__version__"]

if TYPE_CHECKING:
    from regulus.perturb.inference import RegulusModel


def __getattr__(name: str):
    """Delay heavy runtime imports until ``RegulusModel`` is requested."""
    if name == "RegulusModel":
        from regulus.perturb.inference import RegulusModel

        return RegulusModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
