"""Virtual CFO manipulation."""

from regulus.manipulate.core import run_manipulation
from regulus.manipulate.resolver import build_cfo_lookup, resolve_cfo_targets

__all__ = ["build_cfo_lookup", "resolve_cfo_targets", "run_manipulation"]
