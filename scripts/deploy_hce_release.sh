#!/usr/bin/env bash
# Deploy a full HCE release (engine + bot wrapper) to Umbrel without cutting
# games short: stage and build next to the live release, smoke-test the new
# binary, wait until Lichess reports no active games, then switch and restart.
# Rolls back the symlink and chessbot.env if the service fails to come up.
#
# usage: deploy_hce_release.sh HOST [REMOTE_ROOT]
# env:   SYZYGY_PATH (remote dir, default $REMOTE_ROOT/shared/syzygy)
#        HASH_MB     (engine hash per game, default 64)
set -euo pipefail
HOST="${1:?usage: deploy_hce_release.sh HOST [REMOTE_ROOT]}"
REMOTE_ROOT="${2:-/home/umbrel/vidvuds-lab/chess/chessbot}"
[[ "$REMOTE_ROOT" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo 'Unsupported remote path' >&2; exit 1; }
SYZYGY_PATH="${SYZYGY_PATH:-$REMOTE_ROOT/shared/syzygy}"
HASH_MB="${HASH_MB:-64}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_SHORT="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain -- src scripts)" ]]; then
  echo 'Refusing to deploy uncommitted engine/bot changes' >&2
  exit 1
fi
RELEASE_ID="$(date -u +%Y%m%d_%H%M%S)_${GIT_SHORT}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/engine/src"
cp "$REPO_ROOT/src/core/engine/Makefile" "$STAGE/engine/Makefile"
rsync -a --prune-empty-dirs --include='*/' --include='*.c' --include='*.h' \
  --include='*.inc' --include='LICENSE' --include='UPSTREAM_COMMIT' --exclude='*' \
  "$REPO_ROOT/src/core/engine/" "$STAGE/engine/src/"
cp "$REPO_ROOT/src/core/bot/run.py" "$STAGE/run.py"
cat > "$STAGE/BUILD_INFO.txt" <<EOF
release=$RELEASE_ID
built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD)
git_short=$GIT_SHORT
git_branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)
git_dirty=clean
EOF

ssh "$HOST" bash -s -- "$REMOTE_ROOT" "$RELEASE_ID" <<'REMOTE'
set -euo pipefail
ROOT="$1"
DEST="$ROOT/releases/$2"
test -L "$ROOT/current"
test ! -e "$DEST"
mkdir "$DEST"
# Keep the live book, requirements and NN model link; replace engine + wrapper.
cp -a "$ROOT/current/." "$DEST/"
rm -rf "$DEST/engine" "$DEST/run.py" "$DEST/BUILD_INFO.txt" "$DEST/RUNTIME_REPAIR.json" "$DEST/__pycache__"
REMOTE
rsync -a "$STAGE/" "$HOST:$REMOTE_ROOT/releases/$RELEASE_ID/"
ssh "$HOST" "cd '$REMOTE_ROOT/releases/$RELEASE_ID/engine' && make -j\"\$(nproc)\" >build.log 2>&1 || { tail -20 build.log; exit 1; }"

ssh "$HOST" python3 - "$REMOTE_ROOT" "$RELEASE_ID" "$SYZYGY_PATH" "$HASH_MB" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time

import chess
import chess.engine
import requests

root = Path(sys.argv[1])
release = root / 'releases' / sys.argv[2]
syzygy, hash_mb = sys.argv[3], int(sys.argv[4])
previous = (root / 'current').resolve()
py_compile.compile(str(release / 'run.py'), doraise=True)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

for name in ('nn_eval.bin', 'opening_book.txt'):
    old, new = previous / name, release / name
    if old.exists() and digest(old) != digest(new):
        raise RuntimeError(f'Preserved artifact changed: {name}')

# Smoke test: options exist, tables load, a middlegame search and a
# tablebase root answer are legal.
tb_count = len(list(Path(syzygy).glob('*.rtbw'))) if Path(syzygy).is_dir() else 0
with chess.engine.SimpleEngine.popen_uci(str(release / 'engine/chess_uci')) as eng:
    for opt in ('SyzygyPath', 'Hash', 'HceKingPst', 'HcePasser'):
        if opt not in eng.options:
            raise RuntimeError(f'new engine lacks option {opt}')
    cfg = {'Hash': hash_mb, 'Threads': 1}
    if tb_count:
        cfg['SyzygyPath'] = syzygy
    eng.configure(cfg)
    b = chess.Board('r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 9')
    if eng.play(b, chess.engine.Limit(time=1.0)).move not in b.legal_moves:
        raise RuntimeError('illegal middlegame move')
    k = chess.Board('8/8/8/4k3/8/8/8/R3K3 w - - 0 1')
    r = eng.play(k, chess.engine.Limit(time=0.5), info=chess.engine.INFO_SCORE)
    if r.move not in k.legal_moves:
        raise RuntimeError('illegal tablebase move')
    print(f'smoke ok: tables={tb_count} KRK score={r.info.get("score")}', flush=True)

env_path = root / 'chessbot.env'
env_text = env_path.read_text()
env = {}
for line in env_text.splitlines():
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

deadline = time.monotonic() + 3600
while True:
    games = playing()
    if not games:
        time.sleep(3)
        if not playing():
            break
    if time.monotonic() >= deadline:
        raise RuntimeError('Games still active after an hour; staged release left inactive')
    print(f'Waiting for games to finish: {len(games)}', flush=True)
    time.sleep(10)

new_env = [l for l in env_text.splitlines()
           if not l.startswith(('LICHESS_BOT_SYZYGY_PATH=', 'LICHESS_BOT_HASH_MB='))]
if tb_count:
    new_env.append(f'LICHESS_BOT_SYZYGY_PATH={syzygy}')
new_env.append(f'LICHESS_BOT_HASH_MB={hash_mb}')

def point_to(target):
    tmp = root / 'current.release-next'
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(target)
    os.replace(tmp, root / 'current')

try:
    env_path.write_text('\n'.join(new_env) + '\n')
    point_to('releases/' + release.name)
    subprocess.run(['systemctl', '--user', 'restart', 'chessbot.service'], check=True)
    time.sleep(8)
    subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'chessbot.service'], check=True)
except BaseException:
    env_path.write_text(env_text)
    point_to('releases/' + previous.name)
    subprocess.run(['systemctl', '--user', 'restart', 'chessbot.service'], check=True)
    raise

manifest = {'release': str(release), 'previous': str(previous),
            'engine_sha256': digest(release / 'engine/chess_uci'),
            'run_sha256': digest(release / 'run.py'),
            'syzygy_tables': tb_count, 'hash_mb': hash_mb}
(release / 'RELEASE.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
PY
echo "Deployed $RELEASE_ID to $HOST"
