#!/usr/bin/env bash
# Stage this recipe on a fresh node. No training is launched by this script.
set -euo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE=$(cd "$KIT/.." && pwd)
export MINESWEEPER_RUN_ROOT="${MINESWEEPER_RUN_ROOT:-/tmp/$(id -un)-minesweeper}"
source "$KIT/env.sh"
ROOT="$MINESWEEPER_RUN_ROOT"
STAGE_ONLY=0
case "${1:-}" in
    --stage-only) STAGE_ONLY=1 ;;
    '') ;;
    *) echo "Usage: MINESWEEPER_RUN_ROOT=/tmp/... bash $0 [--stage-only]" >&2; exit 1 ;;
esac
for cmd in tar python3 enroot realpath; do command -v "$cmd" >/dev/null; done
if [[ -f "$ROOT/training.pid" ]] && kill -0 "$(cat "$ROOT/training.pid")" 2>/dev/null; then
    echo "A launcher is running in $ROOT; use a fresh directory." >&2
    exit 1
fi
[[ "$ROOT" != "$KIT" ]] || { echo "Choose a separate runtime directory" >&2; exit 1; }
if [[ "$STAGE_ONLY" == 0 ]]; then
    [[ $(uname -m) == x86_64 ]] || { echo "This recipe was tested on x86_64 H100 nodes" >&2; exit 1; }
    command -v nvidia-smi >/dev/null
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi

# Stage exactly this release bundle. Build the replacement first so an
# interrupted copy cannot leave the active source tree half-written.
[[ -f "$BUNDLE/train.py" && -d "$BUNDLE/miles" && -d "$BUNDLE/minesweeper" ]] || {
    echo "setup.sh must be run from the release bundle's runtime directory" >&2
    exit 1
}
NEXT="$ROOT/repo.next"
OLD="$ROOT/repo.old"
rm -rf "$NEXT" "$OLD"
mkdir -p "$NEXT"
tar -C "$BUNDLE" --exclude=.git --exclude='__pycache__' --exclude='*.pyc' -cf - . \
    | tar -C "$NEXT" -xf -
if [[ -d "$ROOT/repo" ]]; then mv "$ROOT/repo" "$OLD"; fi
mv "$NEXT" "$ROOT/repo"
rm -rf "$OLD"

# Keep the conventional sibling path available for older helper scripts.
if [[ -e "$ROOT/minesweeper" && ! -L "$ROOT/minesweeper" ]]; then
    echo "$ROOT/minesweeper exists and is not the managed symlink" >&2
    exit 1
fi
ln -sfn repo/minesweeper "$ROOT/minesweeper"
test -f "$ROOT/minesweeper/minesweeper_env.py"
test -f "$ROOT/minesweeper/game_analysis/turns8/heuristic_not_naive_10k_seeds.txt"
for file in env.sh container.sh training.sh launch.py control.py check_runtime.py; do
    cp "$KIT/$file" "$ROOT/$file"
done
cp "$BUNDLE/README.md" "$ROOT/README.md"
if [[ "$STAGE_ONLY" == 1 ]]; then
    echo "Release source staged at $ROOT; no image import, GPU work, or training performed."
    exit 0
fi

IMAGE="$ROOT/miles-test-20260601.sqsh"
if [[ ! -f "$IMAGE" ]]; then
    enroot import -o "$IMAGE.partial" docker://radixark/miles:test-20260601 \
        2>&1 | tee "$ROOT/logs/enroot-import.log"
    mv "$IMAGE.partial" "$IMAGE"
fi
if [[ ! -d "$ENROOT_DATA_PATH/miles-minesweeper" ]]; then
    enroot create --name miles-minesweeper "$IMAGE" \
        2>&1 | tee "$ROOT/logs/enroot-create.log"
fi

# Match the tested SGLang code. The image already supplies its dependencies.
bash "$ROOT/container.sh" bash -euo pipefail -c '
    cd /sgl-workspace/sglang
    revision=8da71333cbdb131a16b52072f6a67295ec764071
    if [[ $(git rev-parse HEAD) != "$revision" ]]; then
        git fetch origin "$revision"
        git checkout --detach "$revision"
    fi
    test "$(git -C /root/Megatron-LM rev-parse HEAD)" = a381c7da699e4c991f0906434ab2a2c8bd18c1e5
'
bash "$ROOT/container.sh" python3 "$ROOT/check_runtime.py" \
    2>&1 | tee "$ROOT/logs/runtime-check.log"
bash "$ROOT/container.sh" python3 - <<'PY' 2>&1 | tee "$ROOT/logs/model-download.log"
import os
from huggingface_hub import snapshot_download
snapshot_download(
    'Qwen/Qwen3-4B-Instruct-2507',
    revision='cdbee75f17c01a7cc42f958dc650907174af0554',
    local_dir=os.environ['HF_MODEL'],
)
PY
printf '\nSetup complete. Start training with:\npython3 %s/launch.py\n' "$ROOT"
