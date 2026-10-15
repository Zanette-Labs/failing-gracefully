"""Launch one detached run from this node's prepared runtime directory."""
import fcntl
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
lock = (root / 'launch.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
pid_file = root / 'training.pid'
if pid_file.exists():
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        pass
    else:
        raise SystemExit('Launcher is already running; refusing a duplicate.')
job_file = root / 'active_job_id'
if job_file.exists():
    try:
        with urllib.request.urlopen(
            f'http://127.0.0.1:18275/api/jobs/{job_file.read_text().strip()}', timeout=5
        ) as response:
            job = json.load(response)
        if job['status'] in ('PENDING', 'RUNNING'):
            raise SystemExit('Ray job is still active; refusing a duplicate.')
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
    except urllib.error.URLError:
        pass

env = dict(os.environ, MINESWEEPER_RUN_ROOT=str(root))
seed = sys.argv[1] if len(sys.argv) > 1 else '72'
with (root / 'logs/training.log').open('ab', buffering=0) as log:
    process = subprocess.Popen(
        ['bash', str(root / 'container.sh'), 'bash', str(root / 'training.sh'), seed],
        env=env, cwd=root, stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True,
    )
pid_file.write_text(f'{process.pid}\n')
print(f'Launcher PID: {process.pid}\nLog: {root / "logs/training.log"}')
