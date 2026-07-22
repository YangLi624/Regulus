"""CLI help smoke tests that require neither a GPU nor checkpoints."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def test_regulus_main_help():
    r = subprocess.run(
        [sys.executable, "-m", "regulus.cli.main", "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    assert "predict" in r.stdout


@pytest.mark.parametrize(
    "cmd",
    ["predict", "download", "preprocess", "explain", "manipulate", "perturb-train", "graph-train"],
)
def test_subcommand_help(cmd):
    r = subprocess.run(
        [sys.executable, "-m", "regulus.cli.main", cmd, "-h"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
