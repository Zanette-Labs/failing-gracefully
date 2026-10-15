"""Inspect or stop only the Ray job recorded in this runtime directory."""
import json
import re
import sys
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
action = sys.argv[1] if len(sys.argv) > 1 else 'status'
if action not in ('status', 'stop'):
    raise SystemExit('Usage: python3 control.py [status|stop]')
job_id = (root / 'active_job_id').read_text().strip()
url = f'http://127.0.0.1:18275/api/jobs/{job_id}'
request = urllib.request.Request(
    url + '/stop', data=b'{}', headers={'Content-Type': 'application/json'}
) if action == 'stop' else url
with urllib.request.urlopen(request, timeout=10) as response:
    data = json.load(response)
print('Job:', job_id)
print(data if action == 'stop' else {key: data.get(key) for key in ('status', 'message')})
if action == 'status':
    for line in (root / 'logs/training.log').read_text(errors='replace').splitlines()[-8:]:
        print(re.sub(r'\x1b\[[0-9;]*m', '', line)[:500])
