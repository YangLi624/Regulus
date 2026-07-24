# Example input

`example_cells.h5ad` is a deterministic synthetic dataset for checking the
Regulus command-line and Python interfaces. It contains eight cells and the
20,094-gene universe used by `regulus-graph-v1`.

The matrix contains seeded, library-size-normalized synthetic values. It is not
derived from a biological experiment and must not be used to evaluate model
accuracy, infer a mechanism, or support a biological conclusion.

Run the end-to-end example from the repository root:

```bash
bash examples/01_download_bundle.sh
bash examples/02_preprocess_and_predict.sh
bash examples/03_explain.sh
bash examples/04_manipulate_secretion.sh
```

File dimensions, generation parameters, and SHA-256 checksum are recorded in
`example_cells.metadata.json`.
