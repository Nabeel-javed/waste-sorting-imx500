#!/usr/bin/env bash
# Run the waste-sorting demo ON the Sony IMX500 accelerator (not the Pi CPU).
# Intended to live on the Pi at /home/pi/waste-sorting-project/run_imx500_camera.sh
# and be run from the Raspberry Pi desktop / VNC terminal.
#
# Usage:
#   ./run_imx500_camera.sh            # scan mode: banner shows object + target bin
#   ./run_imx500_camera.sh --zones    # older three-column bin-zone overlay
#   ./run_imx500_camera.sh --headless # no window; prints detections
#   Extra args are passed through, e.g. --conf 0.15
set -euo pipefail

PROJECT="/home/pi/waste-sorting-project"
RPK="$PROJECT/models/exported/imx/rpk_out/network.rpk"
LABELS="$PROJECT/models/exported/imx/labels.txt"

# Environment for the on-Pi Wayland/X display (needed for the preview window).
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/1000}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"

if [[ ! -f "$RPK" ]]; then
  echo "network.rpk not found at $RPK" >&2
  echo "Package it first (from your Mac): scripts/deploy_to_pi.sh" >&2
  exit 1
fi

cd "$PROJECT"
# System python3 (picamera2 + IMX500). No torch/ultralytics: inference is on the camera.
exec python3 src/imx500_camera_demo.py \
  --model "$RPK" \
  --labels "$LABELS" \
  --conf 0.25 \
  "$@"
