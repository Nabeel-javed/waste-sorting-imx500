#!/usr/bin/env bash
# Export models/best.pt to Sony IMX500 format (packerOut.zip) using Docker.
#
# Sony's converter toolchain is x86_64-Linux only, so on an Apple Silicon Mac
# this runs an amd64 container under emulation. Run from the repo root.
#
# Usage:
#   scripts/export_imx500_docker.sh [MODEL] [DATA] [IMGSZ]
# Defaults:
#   MODEL=models/best.pt  DATA=dataset/data.yaml  IMGSZ=640
#
# Output: models/best_imx_model/{packerOut.zip,labels.txt,...}
set -euo pipefail

MODEL="${1:-models/best.pt}"
DATA="${2:-dataset/data.yaml}"
IMGSZ="${3:-640}"
IMAGE="waste-imx-export"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building export image ($IMAGE) [amd64 under emulation on Apple Silicon]..."
docker build --platform linux/amd64 -f docker/Dockerfile.imx500-export -t "$IMAGE" .

echo "==> Exporting $MODEL to IMX500 format (imgsz=$IMGSZ)..."
echo "    INT8 quantization runs over calibration images; under emulation this takes ~10-15 min."
docker run --rm --platform linux/amd64 \
  -e YOLO_CONFIG_DIR=/tmp/Ultralytics \
  -v "$REPO_ROOT":/work -w /work \
  "$IMAGE" \
  bash -lc "yolo export model='$MODEL' format=imx data='$DATA' imgsz=$IMGSZ"

echo ""
echo "==> Done. Artifacts in models/best_imx_model/"
echo "    Next: scripts/deploy_to_pi.sh  (copies artifacts to the Pi and runs imx500-package)"
