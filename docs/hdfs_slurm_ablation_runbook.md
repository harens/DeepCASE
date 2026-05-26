# HDFS Slurm Ablation Runbook

This runbook covers the HDFS DeepCASE paper-variant ablation workflow for the maintained `main` branch plus the three paper-aligned experiment branches:

- `exp/hdfs-padding-mask-semantics`
- `exp/hdfs-label-smoothing-frequency`
- `exp/hdfs-decoder-event-hidden-128`
- `exp/hdfs-paper-combined`

## Why worktrees are used

Git worktrees let each variant run in its own checkout without switching branches in a shared directory.

That matters for Slurm because:

- multiple jobs can run at the same time
- a branch switch in one checkout would race with running jobs
- branch-local files, caches, and logs stay isolated
- every result is attributable to a specific branch and commit

The setup here keeps the maintained baseline on the main checkout and creates separate worktrees for the experimental branches.

## Shared environment

Use one shared Python environment for all variants.

The code paths are the same across variants and only the modelling behaviour changes, so separate environments are not required.

Activate your environment before running setup or submitting jobs, and export it into Slurm if your site does not inherit the shell environment by default.
The Slurm script will source the repository virtualenv at `env/bin/activate` via `ablation_paths.env`, so the cluster job uses the same Python environment as the repo checkout.

## Push the local branches

From your local DeepCASE checkout:

```bash
git fetch origin
git push origin \
  main \
  exp/hdfs-padding-mask-semantics \
  exp/hdfs-label-smoothing-frequency \
  exp/hdfs-decoder-event-hidden-128 \
  exp/hdfs-paper-combined
```

If you prefer to push one branch at a time:

```bash
git push origin main
git push origin exp/hdfs-padding-mask-semantics
git push origin exp/hdfs-label-smoothing-frequency
git push origin exp/hdfs-decoder-event-hidden-128
git push origin exp/hdfs-paper-combined
```

## Pull them on the cluster

On the cluster, clone or update the repository and fetch the new refs:

```bash
git clone <your-repo-url> DeepCASE
cd DeepCASE
git fetch origin --prune
git switch main
git pull --ff-only origin main
git switch exp/hdfs-padding-mask-semantics
git pull --ff-only origin exp/hdfs-padding-mask-semantics
git switch exp/hdfs-label-smoothing-frequency
git pull --ff-only origin exp/hdfs-label-smoothing-frequency
git switch exp/hdfs-decoder-event-hidden-128
git pull --ff-only origin exp/hdfs-decoder-event-hidden-128
git switch exp/hdfs-paper-combined
git pull --ff-only origin exp/hdfs-paper-combined
git switch main
```

The experimental branches are then available to the worktree setup script.

If the repository is a fresh clone and only the remote-tracking refs exist, that is fine.
The setup script will create local tracking branches from `origin/<branch>` when needed.

## Configure the worktrees

Pick a shared ablation root on the cluster, preferably on fast shared storage rather than in `$HOME`.

Example:

```bash
./scripts/slurm/setup_hdfs_worktrees.sh
```

If `DEEPCASE_ABLA_ROOT` is unset, the setup script defaults to:

```bash
$PWD/deepcase_hdfs_ablation
```

You can still override it explicitly if you want a different location.

The setup script:

- fetches the latest branch refs
- creates or refreshes the experiment worktrees
- creates `results/hdfs_ablation`, `logs/hdfs_ablation`, and `tmp/hdfs_ablation`
- writes `ablation_paths.env` under the ablation root, including the repo root used to activate `env/bin/activate`

If you rerun it later, it will reset each worktree back to the current branch tip.

## Submit the Slurm array

The Slurm script uses a five-element array:

- `0` -> `main`
- `1` -> `padding-mask-semantics`
- `2` -> `label-smoothing-frequency`
- `3` -> `decoder-hidden-128`
- `4` -> `paper-combined`

Submit it like this:

```bash
export DEEPCASE_ABLA_ROOT="$PWD/deepcase_hdfs_ablation"
source "$DEEPCASE_ABLA_ROOT/ablation_paths.env"
sbatch --chdir="$DEEPCASE_REPO_ROOT" \
  scripts/slurm/run_hdfs_ablation.slurm
```

The job records provenance for each array task and writes results under:

```text
results/hdfs_ablation/<variant>/<commit>/<job_id>_<task_id>/
```

## Inspect outputs

Each run directory contains:

- `metadata.txt`
- `example.log`
- the Slurm stdout/stderr files created by `#SBATCH --output` and `#SBATCH --error`

Useful checks:

```bash
find "$DEEPCASE_ABLA_ROOT/results/hdfs_ablation" -maxdepth 4 -type f | sort
sed -n '1,200p' "$DEEPCASE_ABLA_ROOT/results/hdfs_ablation/main/<commit>/<job_id>_<task_id>/metadata.txt"
```

Replace `<commit>`, `<job_id>`, and `<task_id>` with the values recorded in `metadata.txt`.

## Verify attribution

To confirm a result belongs to the intended variant:

- check `variant=` in `metadata.txt`
- check `branch=` in `metadata.txt`
- check `commit=` in `metadata.txt`
- compare `git status --short --branch` in the metadata with the expected checkout state
- confirm the run directory path includes the same variant and commit hash

The combination of branch name, commit hash, and `git status` is enough to tie a result back to the exact code state that produced it.
