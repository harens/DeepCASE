#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR/../.." rev-parse --show-toplevel)"

: "${DEEPCASE_ABLA_ROOT:?Set DEEPCASE_ABLA_ROOT to a shared cluster path before running this script.}"

WORKTREES_ROOT="$DEEPCASE_ABLA_ROOT/worktrees"
RESULTS_ROOT="$DEEPCASE_ABLA_ROOT/results/hdfs_ablation"
LOGS_ROOT="$DEEPCASE_ABLA_ROOT/logs/hdfs_ablation"
TMP_ROOT="$DEEPCASE_ABLA_ROOT/tmp/hdfs_ablation"
ENV_FILE="$DEEPCASE_ABLA_ROOT/ablation_paths.env"

mkdir -p "$WORKTREES_ROOT" "$RESULTS_ROOT" "$LOGS_ROOT" "$TMP_ROOT"

git -C "$REPO_ROOT" fetch --all --prune

branches=(
  "padding-mask-semantics exp/hdfs-padding-mask-semantics"
  "label-smoothing-frequency exp/hdfs-label-smoothing-frequency"
  "decoder-hidden-128 exp/hdfs-decoder-event-hidden-128"
  "paper-combined exp/hdfs-paper-combined"
)

for entry in "${branches[@]}"; do
  variant="${entry%% *}"
  branch="${entry#* }"
  worktree="$WORKTREES_ROOT/$variant"
  source_ref=""

  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
    source_ref="$branch"
  elif git -C "$REPO_ROOT" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    source_ref="origin/$branch"
  else
    echo "Missing branch: $branch" >&2
    echo "Expected a local branch or origin/$branch after 'git fetch origin --prune'." >&2
    echo "Push the branch from the maintainer clone, then rerun this setup script." >&2
    exit 1
  fi

  if [[ -d "$worktree" ]]; then
    if ! git -C "$worktree" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      echo "Path exists but is not a git worktree: $worktree" >&2
      exit 1
    fi

    git -C "$worktree" checkout -f "$branch"
    git -C "$worktree" reset --hard "$source_ref"
  else
    if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/$branch"; then
      git worktree add "$worktree" "$branch"
    else
      git worktree add -b "$branch" "$worktree" "$source_ref"
    fi
  fi
done

cat > "$ENV_FILE" <<EOF
export DEEPCASE_ABLA_ROOT="$DEEPCASE_ABLA_ROOT"
export DEEPCASE_ABLA_WORKTREES_ROOT="$WORKTREES_ROOT"
export DEEPCASE_ABLA_RESULTS_ROOT="$RESULTS_ROOT"
export DEEPCASE_ABLA_LOGS_ROOT="$LOGS_ROOT"
export DEEPCASE_ABLA_TMP_ROOT="$TMP_ROOT"
EOF

echo "Configured DeepCASE HDFS ablation roots:"
echo "  DEEPCASE_ABLA_ROOT=$DEEPCASE_ABLA_ROOT"
echo "  worktrees=$WORKTREES_ROOT"
echo "  results=$RESULTS_ROOT"
echo "  logs=$LOGS_ROOT"
echo "  tmp=$TMP_ROOT"
echo "  env_file=$ENV_FILE"
