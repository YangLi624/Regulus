# SLURM jobs

| Script | Purpose |
|---|---|
| `train_graph.sbatch` | Train the Regulus graph model through the CLI |
| `train_perturb.sbatch` | Train a perturbation model through the CLI |

Submit from the repository root so that relative config and output paths are
resolved consistently. Activate an environment containing Regulus before
submitting. Add the partition, account, quality-of-service, and time directives
required by your cluster.
