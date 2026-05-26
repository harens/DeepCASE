#!/usr/bin/env bash
#SBATCH --job-name=deepcase-hdfs-ablation
#SBATCH --array=0-4
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

: "${DEEPCASE_ABLA_ROOT:?Set DEEPCASE_ABLA_ROOT to the shared cluster ablation root before submitting this job.}"

VARIANTS=(
  "main"
  "padding-mask-semantics"
  "label-smoothing-frequency"
  "decoder-hidden-128"
  "paper-combined"
)

case "${SLURM_ARRAY_TASK_ID:-}" in
  0) variant="${VARIANTS[0]}" ;;
  1) variant="${VARIANTS[1]}" ;;
  2) variant="${VARIANTS[2]}" ;;
  3) variant="${VARIANTS[3]}" ;;
  4) variant="${VARIANTS[4]}" ;;
  *)
    echo "Unsupported SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 1
    ;;
esac

case "$variant" in
  main)
    MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$PWD}"
    if ! git -C "$MAIN_REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "MAIN_REPO_ROOT is not a git repo: $MAIN_REPO_ROOT" >&2
      echo "Submit from repo root or export MAIN_REPO_ROOT=/path/to/main/repo" >&2
      exit 1
    fi
    worktree="$(git -C "$MAIN_REPO_ROOT" rev-parse --show-toplevel)"
    branch="main"
    ;;
  padding-mask-semantics)
    branch="exp/hdfs-padding-mask-semantics"
    worktree="$DEEPCASE_ABLA_ROOT/worktrees/padding-mask-semantics"
    ;;
  label-smoothing-frequency)
    branch="exp/hdfs-label-smoothing-frequency"
    worktree="$DEEPCASE_ABLA_ROOT/worktrees/label-smoothing-frequency"
    ;;
  decoder-hidden-128)
    branch="exp/hdfs-decoder-event-hidden-128"
    worktree="$DEEPCASE_ABLA_ROOT/worktrees/decoder-hidden-128"
    ;;
  paper-combined)
    branch="exp/hdfs-paper-combined"
    worktree="$DEEPCASE_ABLA_ROOT/worktrees/paper-combined"
    ;;
esac

if ! git -C "$worktree" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Missing worktree for variant '$variant': $worktree" >&2
  echo "Run scripts/slurm/setup_hdfs_worktrees.sh first." >&2
  exit 1
fi

commit="$(git -C "$worktree" rev-parse HEAD)"
job_id="${SLURM_JOB_ID:-local}"
task_id="${SLURM_ARRAY_TASK_ID:-local}"

RESULTS_ROOT="$DEEPCASE_ABLA_ROOT/results/hdfs_ablation"
LOGS_ROOT="$DEEPCASE_ABLA_ROOT/logs/hdfs_ablation"
TMP_ROOT="$DEEPCASE_ABLA_ROOT/tmp/hdfs_ablation"
RUN_DIR="$RESULTS_ROOT/$variant/$commit/${job_id}_${task_id}"
TMP_DIR="$TMP_ROOT/$variant/${job_id}_${task_id}"

mkdir -p "$RUN_DIR" "$LOGS_ROOT" "$TMP_DIR"

export TMPDIR="$TMP_DIR/tmp"
export XDG_CACHE_HOME="$TMP_DIR/cache"
export TORCH_HOME="$TMP_DIR/torch"
export PYTHONPYCACHEPREFIX="$TMP_DIR/pycache"
export MPLCONFIGDIR="$TMP_DIR/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR"

{
  printf 'variant=%s\n' "$variant"
  printf 'branch=%s\n' "$branch"
  printf 'commit=%s\n' "$commit"
  printf 'worktree=%s\n' "$worktree"
  printf 'slurm_job_id=%s\n' "$job_id"
  printf 'slurm_array_task_id=%s\n' "$task_id"
  printf 'hostname=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'python_path=%s\n' "$(command -v python)"
  printf 'git_status_begin\n'
  git -C "$worktree" status --short --branch
  printf 'git_status_end\n'
  printf 'cuda_diagnostics_begin\n'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    printf 'nvidia-smi unavailable\n'
  fi
  python - <<'PY'
import sys

import torch

print(f"python_executable={sys.executable}")
print(f"torch_version={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"device_{index}={torch.cuda.get_device_name(index)}")
PY
  printf 'cuda_diagnostics_end\n'
} > "$RUN_DIR/metadata.txt"

cd "$worktree"

echo "Running variant=$variant branch=$branch commit=$commit"
echo "Results directory: $RUN_DIR"
python example/example_hdfs.py 2>&1 | tee "$RUN_DIR/example.log"
