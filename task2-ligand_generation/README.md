# Task 2 · de novo ligand generation

VoxBind (walk-jump sampling over a voxel denoiser), optionally conditioned on a frozen
density encoder from `pretrain/exps/` or `model_zoo/`.

```
train.py, train_ddp.py        denoiser training (single GPU / DDP)
sample.py, sample_from_file.py   walk-jump sampling for the CrossDocked split / a given pocket
configs/                      Hydra configs; generator recipes are experiment presets (configs/README.md)
evaluation/                   eval.py and one module per metric: quality, vina, posecheck, posebusters;
                              TargetDiff's evaluation code is vendored in evaluation/targetdiff/
scripts/                      01_downstream_voxbind.sh (train), 02_sample.sh and 72_sample_8gpu.sh
                              (sample), 73_evaluate_samples.sh (evaluate), 89_/95_ distribution and
                              pocket-uncertainty analyses, tools/, workflows/sample_4gpu.sh
test/                         fusion-mode tests, encoder-alignment (REPA) probes
notebook/                     pose viewers for generated samples
exps/                         generator runs (not in git)
```

The entry points and configs use paths relative to this folder (`../dataset/data`,
`../pretrain/exps/<run>`, `../model_zoo/<name>`), so run them from here:

```bash
cd task2-ligand_generation
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py smooth_sigma=0.9
python sample.py pretrained_path=exps/exp_sig0.9 wjs.split=test wjs.n_samples_per_pocket=10
```

The launchers in `scripts/` resolve their own paths and run from anywhere, e.g.
`bash task2-ligand_generation/scripts/01_downstream_voxbind.sh --experiment <preset> --gpus 4-7`.
Evaluate a sampling run in the `voxeval` environment (Vina 1.2.2, PoseCheck, PoseBusters; see
`env/README.md`). The defaults are the paper's protocol: the whole receptor, exhaustiveness 32 and
the 79 pockets with density.

```bash
cd task2-ligand_generation
python -m evaluation.eval exps/<run>/samples/<sampling> --metrics quality,vina,posecheck,posebusters
python -m evaluation.eval <out_dir> --meta <baseline>.pt     # a baseline in TargetDiff's meta format
```

Each target gets `eval.json`; the run gets `eval_results.json`, whose `table3` holds the Table 3
numbers. `bash script/05_evaluate.sh` runs the same with the environments resolved for you.
