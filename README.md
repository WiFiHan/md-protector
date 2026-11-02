# MD-ProTector

Code for the paper [MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection](https://arxiv.org/abs/2608.10459).

## Requirements

```bash
conda create -n md-protector python=3.11
conda activate md-protector
pip install -r requirements.txt
```

The paper results were obtained on a single B200 GPU with batch size 256. Class hubs are computed per mini-batch, so a different batch size changes the results.

## Data

Put the datasets under `data/`:

```
data/
  MAGE/
    cross_domains_cross_models/
    unseen_domains/unseen_domain_*/
    unseen_models/unseen_model_*/
  M4/
    SubtaskA/
```

- MAGE: https://github.com/yafuly/MAGE
- M4 (SemEval-2024 Task 8, Subtask A): https://github.com/mbzuai-nlp/SemEval2024-task8
- RAID: downloaded automatically from the Hugging Face Hub ([Shengkun/Raid_split](https://huggingface.co/datasets/Shengkun/Raid_split)). It has no validation split, so we hold out 10% of the training set (indices in `splits/raid/`).

## Training

```bash
python train.py --config configs/mage_cdcm.yaml
```

Configs for all settings in the paper are in `configs/`. To run everything:

```bash
bash scripts/run_main.sh       # Table 1
bash scripts/run_unseen.sh     # Table 2
bash scripts/run_ablations.sh  # Table 3
```

Config values can be overridden from the command line, e.g. `--num_prototypes 4`.

The checkpoint with the best validation AvgRec is saved to `runs/<name>_v0/model_best.pth`. After training, the decision threshold is selected on the validation set and the test results are written to the same folder.

## Evaluation

```bash
python train.py --config configs/mage_cdcm.yaml --eval_only --model_path runs/mage_cdcm_v0/model_best.pth
```

Add `--inference weighted` for the weighted prototype score in Appendix B.2.

## Citation

```bibtex
@misc{han2026mdprotector,
  title={MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection},
  author={Jinmo Han and Jimin Hong and Chanyeong Moon and Ju Yeon Kang and Seonuk Kim and Nam Soo Kim},
  year={2026},
  eprint={2608.10459},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2608.10459},
}
```

## License

MIT
