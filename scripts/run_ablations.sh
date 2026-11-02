#!/usr/bin/env bash
# Table 3: ablations on MAGE CDCM. The full model (R = 8, tau = 0.15, unsup-SimCSE-RoBERTa) is mage_cdcm in run_main.sh.
set -euo pipefail
CFG=configs/mage_cdcm.yaml

# Objective
python train.py --config ${CFG} --name mage_cdcm_wo_pp --pp_ablation drop
python train.py --config ${CFG} --name mage_cdcm_pp_to_repulsion --pp_ablation repulsion
python train.py --config ${CFG} --name mage_cdcm_pp_raw --pp_space raw

# Prototype initialization
python train.py --config ${CFG} --name mage_cdcm_random_init --proto_init random

# Number of prototypes per class
for r in 1 2 4 16 32; do
  python train.py --config ${CFG} --name mage_cdcm_r${r} --num_prototypes ${r}
done

# Temperature
for t in 0.07 0.10 0.20 0.50; do
  python train.py --config ${CFG} --name mage_cdcm_tau${t} --tau ${t}
done

# Encoder backbone
for m in roberta-base princeton-nlp/sup-simcse-roberta-base intfloat/e5-base bert-base-uncased \
         princeton-nlp/unsup-simcse-bert-base-uncased BAAI/bge-base-en-v1.5; do
  python train.py --config ${CFG} --name "mage_cdcm_$(basename ${m})" --model_name "${m}"
done
