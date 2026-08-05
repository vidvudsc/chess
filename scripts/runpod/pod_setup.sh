#!/usr/bin/env bash
# RunPod pod-side setup: deps, threats-patched nnue-pytorch loader, test80 data.
# Run on the pod from /workspace after the repo's nn code is extracted to
# /workspace/chess (see launch_train.sh).
set -euo pipefail
cd /workspace
mkdir -p data out init logs

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake git ninja-build zstd
python3 -m pip install -q --upgrade --break-system-packages python-chess numpy

# The current-threats patch only applies to this pinned nnue-pytorch commit.
NNUE_PYTORCH_COMMIT=a7830b2a91d15f6d3214bd21b1a6cc5cf7701b82
if [[ ! -d nnue-pytorch/.git ]]; then
    git clone https://github.com/official-stockfish/nnue-pytorch.git
fi
git -C nnue-pytorch fetch --depth 1 origin "$NNUE_PYTORCH_COMMIT"
git -C nnue-pytorch checkout -q "$NNUE_PYTORCH_COMMIT"
PATCH=/workspace/chess/src/core/bot/nn/v2/nnue_pytorch_current_threats.patch
if git -C nnue-pytorch apply --check "$PATCH" >/dev/null 2>&1; then
    git -C nnue-pytorch apply "$PATCH"
elif git -C nnue-pytorch apply --reverse --check "$PATCH" >/dev/null 2>&1; then
    echo "threats patch already applied"
else
    echo "FATAL: threats patch does not apply" >&2
    exit 1
fi
cmake -S nnue-pytorch/data_loader/cpp -B nnue-pytorch/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build nnue-pytorch/build -j

for m in 01-jan 02-feb 03-mar; do
    f="test80-2024-${m}-2tb7p.min-v2.v6.binpack"
    if [[ -f "data/$f" ]]; then
        echo "have data/$f"
        continue
    fi
    curl -L --fail --retry 10 --retry-delay 10 -C - -o "data/$f.zst" \
        "https://huggingface.co/datasets/linrock/test80-2024/resolve/main/$f.zst?download=true"
    zstd -d --rm -o "data/$f" "data/$f.zst"
done
ls -la data/
echo SETUP_DONE
