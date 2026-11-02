#!/usr/bin/env bash
# Table 2: MAGE leave-one-out settings (10 unseen domains, 7 unseen generator families).
set -euo pipefail

for cfg in configs/mage_unseen_domains/*.yaml configs/mage_unseen_models/*.yaml; do
  python train.py --config "${cfg}" "$@"
done
