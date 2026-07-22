# Regulus model bundle catalog

`catalog.json` is the package-level index of user-facing pretrained models.
Released archives are self-contained and are installed by `regulus download`.
Use `regulus bundle-build` to assemble an archive from a training config,
perturbation checkpoint, graph checkpoint, and graph asset.

The release catalog contains:

- `norman-post-state-v1`: complete Norman single-gene model.
- `schmidt-post-state-v1`: Schmidt CRISPRa model.
- `tian-crispra-post-state-v1`: Tian CRISPRa model.
- `joung-tfatlas-cfo-post-state-v1`: CFO-only manipulation model.

List or install bundles with:

```bash
regulus download
regulus download norman-post-state-v1 --output-dir bundles
```
