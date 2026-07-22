# Installation

Regulus requires Python 3.10 or newer. Install a release wheel or install the
checked-out source tree:

```bash
pip install .
regulus --help
```

For development and tests:

```bash
pip install ".[dev]"
pytest -q
```

Pretrained graph and perturbation weights are distributed as model bundles,
not inside the Python wheel:

```bash
regulus download
regulus download norman-post-state-v1 --output-dir bundles
```
