#!/usr/bin/env bash
# Update the bot wrapper while preserving the exact live engine and NN model.
set -euo pipefail
HOST="${1:?usage: deploy_hce_runtime.sh HOST [REMOTE_ROOT]}"
REMOTE_ROOT="${2:-/home/umbrel/vidvuds-lab/chess/chessbot}"
[[ "$REMOTE_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsupported remote path' >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_ID="$(date -u +%Y%m%d_%H%M%S)_hce_runtime"
REMOTE_RELEASE="$REMOTE_ROOT/releases/$RELEASE_ID"

ssh "$HOST" bash -s -- "$REMOTE_ROOT" "$RELEASE_ID" <<'REMOTE'
set -euo pipefail
ROOT="$1"
DEST="$ROOT/releases/$2"
test -L "$ROOT/current"
test ! -e "$DEST"
mkdir "$DEST"
cp -a "$ROOT/current/." "$DEST/"
printf '%s\n' "runtime_parent=$(readlink "$ROOT/current")" >> "$DEST/BUILD_INFO.txt"
REMOTE
rsync -a "$REPO_ROOT/src/core/bot/run.py" "$HOST:$REMOTE_RELEASE/run.py"

ssh "$HOST" python3 - "$REMOTE_ROOT" "$RELEASE_ID" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time
import requests

root = Path(sys.argv[1])
release = root / 'releases' / sys.argv[2]
previous = (root / 'current').resolve()
py_compile.compile(str(release / 'run.py'), doraise=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Refuse to smuggle a different chess engine, network, or book into a runtime fix.
for old in previous.rglob('*'):
    if old.is_file() and (old.relative_to(previous).parts[0] == 'engine' or
                          old.name in {'nn_eval.bin', 'opening_book.txt'}):
        new = release / old.relative_to(previous)
        if digest(old) != digest(new):
            raise RuntimeError(f'Preserved artifact changed: {old.name}')

env = {}
for line in (root / 'chessbot.env').read_text().splitlines():
    if '=' in line and not line.lstrip().startswith('#'):
        key, value = line.split('=', 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
if env.get('LICHESS_BOT_BACKEND', 'classic') != 'classic':
    raise RuntimeError('This rollout requires the live classic backend')
token = env.get('LICHESS_TOKEN') or env.get('LICHESS_BOT_TOKEN')
if not token:
    raise RuntimeError('Missing bot token')
session = requests.Session()
session.headers['Authorization'] = 'Bearer ' + token

def playing():
    response = session.get('https://lichess.org/api/account/playing', timeout=20)
    response.raise_for_status()
    games = response.json().get('nowPlaying')
    if not isinstance(games, list):
        raise RuntimeError('Invalid playing snapshot; refusing restart')
    return games

deadline = time.monotonic() + 1200
while True:
    games = playing()
    if not games:
        time.sleep(3)
        if not playing():
            break
    if time.monotonic() >= deadline:
        raise RuntimeError('Games still active; staged release left inactive')
    print(f'Waiting for games to finish: {len(games)}', flush=True)
    time.sleep(15)

dropin = Path.home() / '.config/systemd/user/chessbot.service.d/zz-hce-resources.conf'
dropin.parent.mkdir(parents=True, exist_ok=True)
old_dropin = dropin.read_text() if dropin.exists() else None
dropin.write_text('[Service]\nCPUQuota=400%\nCPUWeight=50\nNice=5\n'
                  'KillMode=control-group\nTimeoutStopSec=15\n')
with (release / 'BUILD_INFO.txt').open('a') as f:
    f.write(f'runtime_repair={sys.argv[2]}\nruntime_sha256={digest(release / "run.py")}\n')

def point_to(target):
    tmp = root / 'current.runtime-next'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, root / 'current')

try:
    point_to('releases/' + sys.argv[2])
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'restart', 'chessbot.service'], check=True)
    time.sleep(5)
    subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'chessbot.service'], check=True)
except BaseException:
    point_to('releases/' + previous.name)
    if old_dropin is None:
        dropin.unlink(missing_ok=True)
    else:
        dropin.write_text(old_dropin)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'restart', 'chessbot.service'], check=True)
    raise

manifest = {'release': str(release), 'previous': str(previous),
            'runtime_sha256': digest(release / 'run.py'),
            'engine_sha256': digest(release / 'engine/chess_uci'),
            'nn_preserved': True, 'cpu_quota': '400%', 'nice': 5}
(release / 'RUNTIME_REPAIR.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
