#!/usr/bin/env bash
# RunPod pod-side training: warm-start continuation of the production net
# over test80 Jan+Feb+Mar. The threats loader is CPU-bound (~25k pos/s on a
# 16-vCPU pod, GPU largely idle), so batch counts are wall-clock budget:
# 100k batches x 8192 = 820M rows = ~8.3h. Checkpoints every ~25 minutes.
set -euo pipefail
cd /workspace

BATCHES=${BATCHES:-100000}
SAVE_EVERY=${SAVE_EVERY:-5000}
LR=${LR:-4e-5}
LR_END=${LR_END:-4e-6}
OUT=${OUT:-/workspace/out/bigdata_l055}

test -f init/warmstart.pt || { echo "FATAL: /workspace/init/warmstart.pt missing" >&2; exit 1; }
test -f data/test80-2024-01-jan-2tb7p.min-v2.v6.binpack || { echo "FATAL: data missing" >&2; exit 1; }

python3 -u chess/src/core/bot/nn/v2/train_binpack.py \
  --input-dir /workspace/data \
  --nnue-pytorch-dir /workspace/nnue-pytorch \
  --output-dir "$OUT" \
  --arch linear-head-screlu-halfka-threats-hm-buckets-psqt \
  --feature-set "HalfKAv2_hm+Full_Threats" \
  --feature-dim 128 \
  --hidden-dim 16 \
  --init-checkpoint /workspace/init/warmstart.pt \
  --loader-workers 12 \
  --batch-size 8192 \
  --batches "$BATCHES" \
  --save-every "$SAVE_EVERY" \
  --cp-scale 600 \
  --score-lambda 0.55 \
  --lr "$LR" \
  --lr-end "$LR_END"

for ckpt in "$OUT"/batch_*.pt "$OUT"/best.pt; do
    [[ -f "$ckpt" ]] || continue
    python3 chess/src/core/bot/nn/export_inference.py \
        --checkpoint "$ckpt" --output "${ckpt%.pt}_v17.bin" --per-row-head-scales || true
done
ls -la "$OUT"
touch /workspace/out/.done
echo TRAIN_DONE
