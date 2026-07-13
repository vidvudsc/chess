#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/workspace/chess}
DATA_ROOT=${DATA_ROOT:-/workspace/data/test80-2024}
NNUE_PYTORCH=${NNUE_PYTORCH:-/workspace/nnue-pytorch}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/output/nnue256x32x32}
DATA_FILE=${DATA_FILE:-test80-2024-01-jan-2tb7p.min-v2.v6.binpack}
EXTRA_DATA_FILES=${EXTRA_DATA_FILES:-}
COMPRESSED_FILE="${DATA_FILE}.zst"
ARCH=${ARCH:-bottleneck-head-screlu}
FEATURE_DIM=${FEATURE_DIM:-256}
BOTTLENECK_DIM=${BOTTLENECK_DIM:-32}
HIDDEN_DIM=${HIDDEN_DIM:-32}
BATCH_SIZE=${BATCH_SIZE:-8192}
BATCHES=${BATCHES:-2000}
SAVE_EVERY=${SAVE_EVERY:-500}
CP_SCALE=${CP_SCALE:-400}
SCORE_LAMBDA=${SCORE_LAMBDA:-0.70}
LR=${LR:-0.0008}
LR_END=${LR_END:-0.00008}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-}

mkdir -p "$DATA_ROOT" "$OUTPUT_DIR"

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq cmake git ninja-build zstd
python3 -m pip install -q --upgrade --break-system-packages huggingface_hub python-chess numpy

if ! python3 - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability()
required = f"sm_{major}{minor}"
raise SystemExit(0 if required in torch.cuda.get_arch_list() else 1)
PY
then
    python3 -m pip install -q --upgrade --break-system-packages torch \
        --index-url https://download.pytorch.org/whl/cu128
fi

(
    if [[ ! -d "$NNUE_PYTORCH/.git" ]]; then
        git clone --depth 1 https://github.com/official-stockfish/nnue-pytorch.git "$NNUE_PYTORCH"
    fi
    if ! git -C "$NNUE_PYTORCH" apply --check \
        "$ROOT/src/core/bot/nn/v2/nnue_pytorch_legacy_halfkp.patch" >/dev/null 2>&1; then
        if ! git -C "$NNUE_PYTORCH" apply --reverse --check \
            "$ROOT/src/core/bot/nn/v2/nnue_pytorch_legacy_halfkp.patch" >/dev/null 2>&1; then
            echo "LegacyHalfKP patch is neither applicable nor already applied" >&2
            exit 1
        fi
    else
        git -C "$NNUE_PYTORCH" apply \
            "$ROOT/src/core/bot/nn/v2/nnue_pytorch_legacy_halfkp.patch"
    fi
    cmake -S "$NNUE_PYTORCH/data_loader/cpp" \
        -B "$NNUE_PYTORCH/build" \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$NNUE_PYTORCH/build" --target training_data_loader -j "$(nproc)"
) &
loader_pid=$!

fetch_binpack() {
    local file=$1
    local compressed="${file}.zst"
    if [[ ! -f "$DATA_ROOT/$file" ]]; then
        if [[ ! -f "$DATA_ROOT/$compressed" ]]; then
            hf download linrock/test80-2024 \
                --repo-type dataset \
                --include "$compressed" \
                --local-dir "$DATA_ROOT" \
                --max-workers 8
        fi
        zstd -d -T0 --rm "$DATA_ROOT/$compressed" -o "$DATA_ROOT/$file"
    fi
}

(
    fetch_binpack "$DATA_FILE"
    for extra in $EXTRA_DATA_FILES; do
        fetch_binpack "$extra"
    done
) &
data_pid=$!

wait "$loader_pid"
wait "$data_pid"

extra_input_args=()
for extra in $EXTRA_DATA_FILES; do
    extra_input_args+=(--extra-input "$DATA_ROOT/$extra")
done

init_args=()
if [[ -n "$INIT_CHECKPOINT" ]]; then
    init_args+=(--init-checkpoint "$INIT_CHECKPOINT")
fi

python3 "$ROOT/src/core/bot/nn/v2/train_binpack.py" \
    --input "$DATA_ROOT/$DATA_FILE" \
    "${extra_input_args[@]}" \
    --nnue-pytorch-dir "$NNUE_PYTORCH" \
    --output-dir "$OUTPUT_DIR" \
    "${init_args[@]}" \
    --arch "$ARCH" \
    --feature-dim "$FEATURE_DIM" \
    --bottleneck-dim "$BOTTLENECK_DIM" \
    --hidden-dim "$HIDDEN_DIM" \
    --cp-scale "$CP_SCALE" \
    --score-lambda "$SCORE_LAMBDA" \
    --loss-power 2.5 \
    --batch-size "$BATCH_SIZE" \
    --batches "$BATCHES" \
    --loader-workers 2 \
    --lr "$LR" \
    --lr-end "$LR_END" \
    --weight-decay 0.00001 \
    --save-every "$SAVE_EVERY" \
    --log-every 25 \
    --device cuda
