#!/usr/bin/env bash
# One-shot RunPod NNUE training launch, run from the repo root on the Mac.
#   scripts/runpod/launch_train.sh <warmstart-checkpoint-source>
# where the checkpoint source is an mlpc path pushed separately, e.g.
#   D:\nnue-2800\runs\v17_threats_test80_mix_lambda055\batch_000500.pt
#
# Hard-won specifics (2026-08-05, do not remove):
# - --ports "22/tcp" is REQUIRED or direct ssh is never provisioned and
#   `runpodctl ssh info` reports "pod not ready" forever.
# - Template runpod-torch-v280 is broken; use the pinned image below.
# - Check the account balance covers this run PLUS any other running pods:
#   RunPod force-stops EVERY pod on the account when balance reaches zero.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEY=${RUNPOD_KEY:-$HOME/.runpod/ssh/RunPod-Key-Go}
IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
GPU=${GPU:-"NVIDIA GeForce RTX 4090"}
CLOUD=${CLOUD:-SECURE}
MLPC_CKPT=${1:?"usage: launch_train.sh <mlpc checkpoint path (windows syntax)>"}
WORK=$(mktemp -d)

APIKEY=$(grep "^apikey" ~/.runpod/config.toml | cut -d"'" -f2)
BALANCE=$(curl -s https://api.runpod.io/graphql -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $APIKEY" -d '{"query":"query { myself { clientBalance } }"}' \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['myself']['clientBalance'])")
echo "account balance: \$$BALANCE"
python3 -c "import sys; sys.exit(0 if float('$BALANCE') >= 8 else 1)" || {
  echo "FATAL: balance under \$8 — top up before launching (run needs ~\$6 plus other pods' burn)"; exit 1; }

echo "[1] create pod"
runpodctl pod create --image "$IMAGE" --gpu-id "$GPU" --cloud-type "$CLOUD" \
  --volume-in-gb 100 --container-disk-in-gb 30 --ports "22/tcp" --name nnue-train > "$WORK/create.json"
POD=$(grep -o '"id": *"[a-z0-9]*"' "$WORK/create.json" | head -1 | cut -d'"' -f4)
[ -n "$POD" ] || { echo "FATAL create:"; tail -3 "$WORK/create.json"; exit 1; }
echo "pod=$POD"

echo "[2] wait for ssh"
IP=""; PORT=""
for i in $(seq 1 60); do
  INFO=$(runpodctl ssh info "$POD" 2>/dev/null || true)
  IP=$(echo "$INFO" | grep -o '"ip": *"[^"]*"' | head -1 | cut -d'"' -f4)
  PORT=$(echo "$INFO" | grep -o '"port": *[0-9]*' | head -1 | grep -o '[0-9]*')
  [ -n "$IP" ] && [ -n "$PORT" ] && break
  sleep 15
done
[ -n "$IP" ] || { echo "FATAL: no ssh in 15min"; runpodctl pod delete "$POD"; exit 1; }
echo "ssh root@$IP:$PORT"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=no root@$IP"
SCP="scp -i $KEY -P $PORT -o StrictHostKeyChecking=no"

echo "[3] ship code"
tar -czf "$WORK/code.tar.gz" -C "$ROOT" src/core/bot/nn scripts/decompress_zstd.py
$SCP "$WORK/code.tar.gz" "$ROOT/scripts/runpod/pod_setup.sh" "$ROOT/scripts/runpod/pod_train.sh" "root@$IP:/workspace/"
$SSH 'cd /workspace && mkdir -p chess logs init && tar -xzf code.tar.gz -C chess && echo CODE_OK'

echo "[4] setup (data download; ~15 min)"
$SSH 'cd /workspace && bash pod_setup.sh > logs/setup.log 2>&1 && echo SETUP_OK || { tail -5 logs/setup.log; exit 1; }'

echo "[5] warm-start checkpoint from mlpc"
ssh mlpc "C:\\Windows\\System32\\OpenSSH\\scp.exe -i C:\\Users\\vidvudsml\\.ssh\\RunPod-Key-Go -P $PORT -o StrictHostKeyChecking=no $MLPC_CKPT root@$IP:/workspace/init/warmstart.pt"
$SSH 'python3 -c "import torch; torch.load(\"/workspace/init/warmstart.pt\", map_location=\"cpu\", weights_only=False); print(\"CHECKPOINT_VALID\")"'

echo "[6] launch training"
$SSH 'cd /workspace && (nohup bash pod_train.sh > logs/train.log 2>&1 &) && sleep 60 && tail -3 logs/train.log'
echo "STARTED pod=$POD ip=$IP port=$PORT"
echo "watch:   ssh -i $KEY -p $PORT root@$IP 'tail -f /workspace/logs/train.log'"
echo "cleanup: runpodctl pod delete $POD   (after pulling out/bigdata_l055/*_v17.bin)"
