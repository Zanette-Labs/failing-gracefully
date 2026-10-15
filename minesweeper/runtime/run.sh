#!/usr/bin/env bash
# Prepare and launch a complete eight-GPU run without any writes to host home.
set -euo pipefail
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    cat <<'HELP'
Usage: bash run.sh [seed]

Sets up Enroot, downloads the model, and starts detached eight-GPU training.
Uses 8-turn medium Minesweeper: 9,500 training puzzles and 500 validation puzzles.
Each step generates 32 prompts x 16 rollouts = 512 episodes, then updates once.
Seed defaults to 72. All runtime files use /tmp/$USER-minesweeper.
Set MINESWEEPER_RUN_ROOT=/tmp/another-directory to change that location.
Set GRACEFUL_REWARD=0.5 to change the grace reward (default 0; 0 <= value < 1).
Set ADVANTAGE_ESTIMATOR=rloo or gracefulrl to change the estimator (default maxrl).
Set GRACEFULRL_EPS to change GracefulRL rank smoothing (default 0, the exact estimator).
Set MINESWEEPER_SMOKE_TEST=1 to run one tiny end-to-end optimizer step.
Training runs for 250 steps and saves its checkpoint after step 250 under /tmp.
Existing active runs are left running.
W&B defaults to offline when neither WANDB_API_KEY nor .netrc credentials exist.
The node must already have working Enroot/NVIDIA support and eight available GPUs.
HELP
    exit 0
fi
SEED="${1:-72}"
[[ $# -le 1 && "$SEED" =~ ^[0-9]+$ ]] || { echo "Pass one nonnegative integer seed, or --help" >&2; exit 1; }
export MINESWEEPER_RUN_ROOT="${MINESWEEPER_RUN_ROOT:-/tmp/$(id -un)-minesweeper}"
source "$KIT/env.sh"
ROOT="$MINESWEEPER_RUN_ROOT"
command -v python3 >/dev/null
command -v flock >/dev/null
exec 9>"$ROOT/setup-and-launch.lock"
flock -n 9 || { echo "Setup/launch is already in progress in $ROOT" >&2; exit 1; }

# Check before setup can replace runtime files or modify the writable image.
CHECK_STATUS=0
python3 - "$ROOT" <<'PY' || CHECK_STATUS=$?
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])
pid_file = root / 'training.pid'
if pid_file.exists():
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        print(f'Training launcher already active in {root}.')
        print(f'Log: {root / "logs/training.log"}')
        sys.exit(0)
job_file = root / 'active_job_id'
if job_file.exists():
    job_id = job_file.read_text().strip()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:18275/api/jobs/{job_id}', timeout=5) as response:
            job = json.load(response)
        if job['status'] in ('PENDING', 'RUNNING'):
            print(f'Ray job {job_id} is already {job["status"]}.')
            print(f'Log: {root / "logs/training.log"}')
            sys.exit(0)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    except urllib.error.URLError:
        pass
sys.exit(10)  # No active run; proceed with setup.
PY
case "$CHECK_STATUS" in
    0) exit 0 ;;
    10) ;;
    *) exit "$CHECK_STATUS" ;;
esac

# A fresh node can run without an interactive W&B login.
if [[ -z "${WANDB_MODE:-}" && -z "${WANDB_API_KEY:-}" ]]; then
    if ! python3 - <<'PY'
import netrc
import os
import sys
try:
    credentials = netrc.netrc(os.path.join(os.path.expanduser('~'), '.netrc')).authenticators('api.wandb.ai')
    sys.exit(0 if credentials and credentials[2] else 1)
except (OSError, netrc.NetrcParseError):
    sys.exit(1)
PY
    then
        export WANDB_MODE=offline
        echo "No W&B credentials found; logging offline under $ROOT/wandb."
    fi
fi
export GIT_TERMINAL_PROMPT=0
echo "Preparing eight-GPU Enroot runtime in $ROOT"
bash "$KIT/setup.sh" 2>&1 | tee "$ROOT/logs/setup.log"
if [[ "$MINESWEEPER_SMOKE_TEST" == 1 ]]; then
    echo "Starting end-to-end smoke test (estimator $ADVANTAGE_ESTIMATOR; seed $SEED; grace $GRACEFUL_REWARD; one optimizer step)."
else
    echo "Starting 8-turn medium training (estimator $ADVANTAGE_ESTIMATOR; gracefulrl_eps $GRACEFULRL_EPS; seed $SEED; grace $GRACEFUL_REWARD; 16 rollouts/prompt; 250 steps; checkpoint at step 250)."
fi
exec python3 "$ROOT/launch.py" "$SEED"
