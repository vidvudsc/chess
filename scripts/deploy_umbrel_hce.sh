#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <host> [remote_root]" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="$1"
REMOTE_ROOT="${2:-/home/umbrel/vidvuds-lab/chess/chessbot}"
KEEP_RELEASES="${KEEP_RELEASES:-5}"
LOCAL_NN_MODEL=""

for candidate in \
  "$ROOT_DIR/src/core/bot/nn/model/nn_eval.bin" \
  "$ROOT_DIR/current/nn_eval.bin"; do
  if [[ -f "$candidate" ]]; then
    LOCAL_NN_MODEL="$candidate"
    break
  fi
done

GIT_SHORT="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo manual)"
RELEASE_ID="$(date -u +%Y%m%d_%H%M%S)_${GIT_SHORT}"
STAGE_DIR="$ROOT_DIR/dist/umbrel_hce/releases/$RELEASE_ID"
REMOTE_RELEASE_DIR="$REMOTE_ROOT/releases/$RELEASE_ID"

"$ROOT_DIR/scripts/package_umbrel_hce.sh" "$STAGE_DIR" "$RELEASE_ID"

ssh "$HOST" "mkdir -p '$REMOTE_RELEASE_DIR' '$REMOTE_ROOT/shared' '$REMOTE_ROOT/shared/logs' '$REMOTE_ROOT/releases'"

rsync -avh --delete \
  --exclude '__pycache__/' \
  "$STAGE_DIR/" "$HOST:$REMOTE_RELEASE_DIR/"

if [[ -n "$LOCAL_NN_MODEL" ]]; then
  rsync -avh --checksum "$LOCAL_NN_MODEL" "$HOST:$REMOTE_ROOT/shared/nn_eval.bin"
fi

ssh "$HOST" bash -s -- "$REMOTE_ROOT" "$RELEASE_ID" "$KEEP_RELEASES" <<'EOF'
set -euo pipefail

ROOT="$1"
RELEASE="$2"
KEEP="$3"
RELEASE_DIR="$ROOT/releases/$RELEASE"
MIGRATION_DIR="$ROOT/releases/_migration_$RELEASE"

ensure_migration_dir() {
  mkdir -p "$MIGRATION_DIR"
}

move_if_present() {
  local src="$1"
  local dst_name="$2"
  if [[ -e "$src" || -L "$src" ]]; then
    ensure_migration_dir
    mv "$src" "$MIGRATION_DIR/$dst_name"
  fi
}

mkdir -p "$ROOT/releases" "$ROOT/shared" "$ROOT/shared/logs"

if [[ -e "$ROOT/current" && ! -L "$ROOT/current" ]]; then
  move_if_present "$ROOT/current" "current_pre_symlink"
fi

for name in bots engine run.py requirements.txt __pycache__ .env.example opening_book.txt BUILD_INFO.txt log.txt; do
  move_if_present "$ROOT/$name" "$name"
done

cd "$RELEASE_DIR/engine"
make -j"$(nproc)"

if [[ -f "$RELEASE_DIR/requirements.txt" ]]; then
  python3 -m pip install --user -r "$RELEASE_DIR/requirements.txt"
fi

if [[ -f "$ROOT/shared/nn_eval.bin" ]]; then
  ln -sfn ../../shared/nn_eval.bin "$RELEASE_DIR/nn_eval.bin"
fi

ln -sfn "releases/$RELEASE" "$ROOT/current"

if systemctl --user cat chessbot.service >/dev/null 2>&1; then
  systemctl --user restart chessbot.service
fi

mapfile -t release_dirs < <(
  find "$ROOT/releases" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' -printf '%T@ %P\n' \
    | sort -nr \
    | awk '{print $2}'
)

for idx in "${!release_dirs[@]}"; do
  if (( idx >= KEEP )); then
    rm -rf "$ROOT/releases/${release_dirs[$idx]}"
  fi
done

rmdir "$MIGRATION_DIR" 2>/dev/null || true

echo "release_dir=$RELEASE_DIR"
echo "current_link=$(readlink "$ROOT/current")"
if [[ -f "$RELEASE_DIR/BUILD_INFO.txt" ]]; then
  echo "-- build info --"
  cat "$RELEASE_DIR/BUILD_INFO.txt"
fi
if systemctl --user cat chessbot.service >/dev/null 2>&1; then
  echo "-- service status --"
  systemctl --user status --no-pager chessbot.service || true
fi
EOF

echo "Deployed canonical HCE bundle to $HOST:$REMOTE_ROOT/releases/$RELEASE_ID"
