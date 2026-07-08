#!/usr/bin/env bash
# Copy the IMX500 export artifacts + on-camera script to the Raspberry Pi and
# package packerOut.zip into a network.rpk there (imx500-package is Pi-side only).
#
# Usage:
#   scripts/deploy_to_pi.sh [PI_HOST]
# Default PI_HOST=pi   (uses your ~/.ssh/config "pi" entry)
set -euo pipefail

PI_HOST="${1:-pi}"
PI_PROJECT="/home/pi/waste-sorting-project"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f models/best_imx_model/packerOut.zip ]]; then
  echo "models/best_imx_model/packerOut.zip not found. Run scripts/export_imx500_docker.sh first." >&2
  exit 1
fi

echo "==> Copying export artifacts + on-camera script + run helper to $PI_HOST..."
ssh "$PI_HOST" "mkdir -p $PI_PROJECT/models/exported/imx $PI_PROJECT/src"
scp models/best_imx_model/packerOut.zip models/best_imx_model/labels.txt \
    "$PI_HOST:$PI_PROJECT/models/exported/imx/"
scp src/imx500_camera_demo.py "$PI_HOST:$PI_PROJECT/src/"
scp scripts/run_imx500_camera.sh "$PI_HOST:$PI_PROJECT/run_imx500_camera.sh"
ssh "$PI_HOST" "chmod +x $PI_PROJECT/run_imx500_camera.sh"

echo "==> Packaging network.rpk on the Pi..."
ssh "$PI_HOST" "cd $PI_PROJECT/models/exported/imx && rm -rf rpk_out && \
  imx500-package -i packerOut.zip -o rpk_out && ls -la rpk_out/"

echo ""
echo "==> Done. network.rpk is at $PI_PROJECT/models/exported/imx/rpk_out/network.rpk"
echo "    Run the live demo on the Pi with: $PI_PROJECT/run_imx500_camera.sh"
