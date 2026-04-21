#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${1:-$ROOT_DIR/dist/umbrel_hce/release}"
RELEASE_ID="${2:-manual_$(date -u +%Y%m%d_%H%M%S)}"
ENGINE_STAGE="$STAGE_DIR/engine/src"

git_value() {
  git -C "$ROOT_DIR" "$@" 2>/dev/null || true
}

GIT_COMMIT="$(git_value rev-parse HEAD)"
GIT_BRANCH="$(git_value rev-parse --abbrev-ref HEAD)"
GIT_SHORT="$(git_value rev-parse --short HEAD)"
GIT_STATUS="$(git_value status --porcelain)"
GIT_DIRTY="clean"
if [[ -n "$GIT_STATUS" ]]; then
  GIT_DIRTY="dirty"
fi
BUILD_TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

rm -rf "$STAGE_DIR"
mkdir -p "$ENGINE_STAGE"

rsync -a \
  "$ROOT_DIR/src/core/engine/Makefile" \
  "$STAGE_DIR/engine/Makefile"

rsync -a \
  --include='*.c' \
  --include='*.h' \
  --include='*.inc' \
  --exclude='*' \
  "$ROOT_DIR/src/core/engine/" "$ENGINE_STAGE/"

rsync -a \
  "$ROOT_DIR/src/core/bot/run.py" \
  "$ROOT_DIR/src/core/bot/requirements.txt" \
  "$STAGE_DIR/"

if [[ -f "$ROOT_DIR/src/core/bot/umbrel.env.example" ]]; then
  rsync -a "$ROOT_DIR/src/core/bot/umbrel.env.example" "$STAGE_DIR/.env.example"
fi

if [[ -f "$ROOT_DIR/data/openings/opening_book.txt" ]]; then
  rsync -a "$ROOT_DIR/data/openings/opening_book.txt" "$STAGE_DIR/opening_book.txt"
elif [[ -f "$ROOT_DIR/data/openings/opening_games_100.txt" ]]; then
  rsync -a "$ROOT_DIR/data/openings/opening_games_100.txt" "$STAGE_DIR/opening_book.txt"
fi

cat > "$STAGE_DIR/BUILD_INFO.txt" <<EOF
release=$RELEASE_ID
built_at=$BUILD_TS
git_commit=${GIT_COMMIT:-unknown}
git_short=${GIT_SHORT:-unknown}
git_branch=${GIT_BRANCH:-unknown}
git_dirty=$GIT_DIRTY
EOF

printf 'Packaged Umbrel bundle at %s\n' "$STAGE_DIR"
