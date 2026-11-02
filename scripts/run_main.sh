#!/usr/bin/env bash
# Table 1: MAGE CDCM, RAID and M4.
set -euo pipefail

for cfg in mage_cdcm raid m4; do
  python train.py --config configs/${cfg}.yaml "$@"
done
