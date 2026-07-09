#!/usr/bin/env bash
# Backup demo: run the waste-sorting scan UI on the MacBook webcam.
# Same voting + banner behaviour as the Pi/IMX500 demo, but inference runs
# locally (Apple GPU) with models/best.pt instead of on the camera sensor.
#
# Usage:
#   scripts/run_mac_demo.sh             # scan mode (default)
#   scripts/run_mac_demo.sh --zones     # legacy three-column overlay
#   Extra args pass through, e.g. --conf 0.35 --camera 1
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f models/best.pt ]]; then
  echo "models/best.pt not found - pull it from the training server first." >&2
  exit 1
fi

exec .venv/bin/python src/webcam_demo.py --model models/best.pt --conf 0.35 "$@"
