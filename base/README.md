# `base/`: external baselines

The affinity models CoDE is compared against in Table 1. Each subfolder holds one baseline's
code, either the upstream repository or our reimplementation, plus our glue for our splits
(`_edrscc/` or `src/`). Every baseline has its own conda env and imports nothing from `voxbind/`.

## Running

Every baseline has one script, `scripts/<baseline>.sh`, with two stages run on one or more frozen
splits (`voxbind/splits/<split>.csv`), and `baseline.py` runs any set of them, one baseline per GPU:

| stage | what it does |
|---|---|
| `featurize` | converts our complexes into the baseline's input format, in `base/data/<baseline>/` |
| `train_eval` | trains on the split (seeds 0-4), predicts its test set, and CASF-2016 for `lp_edrscc_v2`, then scores the predictions into `base/results/<method>.csv` |

```bash
python base/baseline.py --list                                   # baselines and their scripts
python base/baseline.py --status                                 # what base/results/ holds
python base/baseline.py --baselines get,profsa --dry-run         # print the exact commands
python base/baseline.py --all --gpus 4,5,6,7                     # every baseline, train_eval
python base/baseline.py --all --eval-only                        # rescore the predictions on disk
bash base/scripts/cheapnet.sh featurize --split lp_edrscc_v2     # one baseline by hand
bash base/scripts/cheapnet.sh train_eval --split lp_edrscc_v2_cl123
```

Defaults: splits `lp_edrscc_v2 lp_edrscc_v2_cl123`, seeds 0-4, GPUs 4-7 (0-3 is the
pretraining lane). A script prints its usage when run without a stage; `DRY_RUN=1` prints its
commands, `EVAL_ONLY=1` skips training. Logs go to `log/baselines/<timestamp>/`.

**Data.** Every baseline reads the complexes, labels and splits our encoders use, from the paths in
`datasource.py`: the PDBbind structures under `dataset/data/pdbbind/structures/`,
`LP_PDBBind.csv` for pK, and `voxbind/splits/`. What it featurises goes to `base/data/<baseline>/`
(not in git).

**Results.** `evaluate.py` reads each baseline's per-complex predictions, joins them to our labels
and scores them on the test sets the encoders are scored on, writing one row per split, test set
and seed to `base/results/<method>.csv` (n, Pearson r, Spearman ρ, RMSE, and the prediction file).
The predictions themselves are copied to `base/results/preds/<method>/`.

| model trained on | test set | complexes | pids (`voxbind/splits/`) |
|---|---|---:|---|
| `lp_edrscc_v2` | FULL | 1,320 | the split's test set |
| | CASF-nontrain | 124 | `casf2016_nontrain_cohort.csv` |
| | CASF-clean | 92 | `casf2016_clean_cohort.csv` |
| `lp_edrscc_v2_cl123` | CL3 | 733 | the split's test set |
| | CL3-ID60 | 454 | `cl123_test_novel60_cohort.csv` |
| | CL3-ID30 | 262 | `cl123_test_novel30_cohort.csv` |

## Baselines

| baseline | paper | protocol | script | env |
|---|---|---|---|---|
| [AEV-PLIG](https://github.com/oxpig/AEV-PLIG) | [Commun. Chem. 2025](https://www.nature.com/articles/s42004-025-01428-y) | trained on our splits | `aevplig.sh` | aevplig |
| [BindNet](https://github.com/fengshikun/BindNet) | ICLR 2024 | trained from scratch (released weights restricted) | `bindnet.sh` | bindnet |
| [CheapNet](https://github.com/hyukjunlim/CheapNet) | [ICLR 2025](https://openreview.net/forum?id=A1HhtITVEi) | trained on our splits | `cheapnet.sh` | cheapcross |
| [DSMBind](https://github.com/wengong-jin/DSMBind) | bioRxiv 2023 | zero-shot energy, \|r\| and \|ρ\| | `dsmbind.sh` | dsmbind |
| DeepDTA | [Bioinformatics 2018](https://doi.org/10.1093/bioinformatics/bty593) | trained, sequence + SMILES (DeepPurpose) | `dta.sh` | dta |
| MolTrans | [Bioinformatics 2021](https://doi.org/10.1093/bioinformatics/btaa880) | trained, sequence + SMILES (DeepPurpose) | `dta.sh` | dta |
| [GET](https://github.com/THUNLP-MT/GET) | [ICML 2024](https://arxiv.org/abs/2306.01474) | trained on our splits | `get.sh` | get |
| EGNN, EGNN + TargetDiff | [TargetDiff, ICLR 2023](https://openreview.net/forum?id=kJqXEPXMsE0) | trained with TargetDiff's property-prediction code; + TargetDiff adds the hidden states of its pretrained diffusion model | `targetdiff.sh` | decompdiff |
| [GeoSSL-DDM](https://github.com/chao1224/GeoSSL) | [arXiv 2206.13602](https://arxiv.org/abs/2206.13602) | frozen encoder + MLP probe | `geossl.sh` | aevplig |
| HBGSA | [arXiv 2604.23115](https://arxiv.org/abs/2604.23115) | our reimplementation, trained | `hbgsa.sh` | hbgsa |
| HonestAffinity | [arXiv 2606.03422](https://arxiv.org/abs/2606.03422) | our reimplementation, trained | `honestaffinity.sh` | dsmbind |
| IPNet (frozen, scratch) | [IPDiff, ICLR 2024](https://openreview.net/forum?id=qH9nrMNTIW) | frozen + MLP head; trained from scratch | `ipnet.sh` | get |
| [Nesso-1](https://github.com/recursionpharma/nesso) | Recursion 2026 | zero-shot co-folding; trained on our test sets | `nesso.sh` | nesso |
| [ProFSA](https://github.com/bowen-gao/ProFSA) | [ICLR 2024](https://arxiv.org/abs/2310.07229) | frozen encoder + regression head | `profsa.sh` | profsa |

`decompdiff/` is the DecompDiff generative baseline of task 2, sampled and scored there.

Until 2026-10-10 the two EGNN rows came from the GET repo (`get.sh`, methods `egnn` and `egnn_td`), and
"EGNN + TargetDiff" there was the GET repo's EGNN with N(0, 1 Å) coordinate noise in training and
inference, not TargetDiff's features. `targetdiff.sh` (methods `td_egnn`, `td_egnn_enc`) replaces both
with TargetDiff's own code: `targetdiff/` is the upstream repository at a pinned commit, with the two
changes listed in `targetdiff/PATCHES.md`. Both models have one output head per measurement type
(Ki, Kd, IC50) and use the complex's type to pick it, as TargetDiff does.

## From predictions to scores

`evaluate.py` writes `base/results/<method>.csv` for the six test sets, as mean ± population std over
seeds; the other numbers come from each baseline's own result JSONs and the cross-baseline scorers in
`_casf/`:

| numbers | scorer | reads |
|---|---|---|
| Table 1 (FULL, CL3), Table 1b (CL3 ID60/ID30), CASF non-train and clean | `evaluate.py` → `results/<method>.csv` | each baseline's predictions |
| Table 1 CL1 and CL12 tiers | each baseline's result JSON | its training run |
| 90% CIs of the CASF tables | `_casf/bootstrap_casf_ci.py` → `casf_bootstrap_ci.json` (shown while its mean matches the CSV) | each baseline's CASF predictions |
| CASF clean ID60/ID30 (Table A2) | `task1-affinity_regression/evals/casf_similarity_rescore.py` → `casf_similarity/` | CASF predictions |
| LP-novelty appendix | `_casf/score_novel_subsets.py` → `novel_subsets/` | per-complex v2 test predictions |

## Other folders

| path | what |
|---|---|
| `scripts/`, `baseline.py`, `evaluate.py`, `datasource.py` | the run scripts, their runner, the scorer, the data paths |
| `data/`, `results/` | featurised inputs and scores (not in git) |
| `models/` | every trained model as `models/<method>/<split>/seed<N>/`: the checkpoint under its trainer's name, the config and `cmd.txt` (command and git commit); the CASF-2016 model of a seed in `casf/`. Written by `model_store.py` (not in git). Checkpoints of 64 MB or more are hard links to the trainer's file. |
| `_casf/` | the cross-baseline scorers above, their inputs and outputs |
| `_tools/mmseqs/` | MMseqs2 for the sequence-similarity scripts in `task1-affinity_regression/`; not in git: `wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz && tar xzf mmseqs-linux-avx2.tar.gz -C base/_tools` (the copy used reports `mmseqs version` 17b688d21dda) |
| `../task1-affinity_regression/test/` | the external test cohorts (CASP16, BDB2020+, ChEMBL35–PLINDER30, OpenBind), moved there on 2026-10-09; not in git |

## Before re-running for the paper

Before the report switched to `base/results/` (2026-10-09), those CSVs reproduced its Table 1
numbers exactly for 11 baselines. The other five differ because the predictions on disk come from
earlier runs than the metrics reported then: BindNet, HonestAffinity, IPNet (frozen and scratch)
and ProFSA have lp_edrscc_v2 test predictions for 3 seeds only, BindNet's CL3 seed 0 comes from
another checkpoint, and IPNet's CL3 predictions come from separately trained heads. `train_eval`
now saves the predictions of the very models it scores, so rerunning it fixes all five.

- GET, EGNN and EGNN + TargetDiff: the current CASF predictions came from the first checkpoint
  `find` listed, which is not the best-validation checkpoint for GET seeds 0/2/4, EGNN seed 0
  and EGNN + TargetDiff seed 4. `get.sh` now uses the best one.
- DeepDTA and MolTrans: their CL3 predictions were regenerated after the last
  `fill_cl123_novelty.py` run, so the old Table 1b was stale for them; the report now scores them
  from `base/results/`.
- DSMBind: a full `score_novel_subsets.py` run overwrites its zero-shot novelty files with
  probe predictions (`novel_preds/DSMBind_v2_seed*.csv`), and so does `casf_similarity_rescore.py`
  for its Table A2 files.
- The CL1 and CL12 split CSVs are in `legacy/voxbind/splits/`; copy them back before running
  those tiers.
