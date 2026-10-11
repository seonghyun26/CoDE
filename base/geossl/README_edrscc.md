# GeoSSL baseline (frozen pretrained encoder + probe) — lp_edrscc_v2 & seq-id splits

Evaluate **GeoSSL-DDM** (Liu et al., *Molecular Geometry Pretraining with SE(3)-Invariant
Denoising Distance Matching*, 2022; [arXiv 2206.13602](https://arxiv.org/abs/2206.13602);
[repo](https://github.com/chao1224/GeoSSL)) as an affinity baseline. GeoSSL-DDM self-supervises
a **3D SchNet** encoder on single small molecules (Molecule3D) by denoising pairwise-distance
matching. We take the **released pretrained encoder frozen** and train only a 2-layer MLP head
on our splits — the same frozen-encoder + probe protocol as VoxBind's own C / C+D+G probes and
the [ProFSA](../profsa) baseline.

> **Not their protocol.** GeoSSL's paper *fine-tunes* the whole SchNet end-to-end on ATOM3D LBA.
> We instead **freeze** it (a probe), so these numbers measure how transferable the
> molecule-pretrained features are out-of-the-box to protein–ligand complexes — directly
> comparable to our frozen encoders, not to GeoSSL's fine-tuned LBA table.

## Method

`schnet.py` is a self-contained re-implementation of `Geom3D/models/schnet.py` (hidden 128,
6 interaction blocks, 51 Gaussians, cutoff 10 Å, 9-class element embedding, mean readout;
0.6 M params). It loads the released **GeoSSL-DDM SchNet** weights with `strict=True`
(`weights/geossl_ddm_schnet.pth`, extracted from the authors' Google-Drive checkpoint bundle,
config `Molecule3D_1000000_schnet/..._30_symmetry_0.5`). Two dependency-free swaps: `radius_graph`
is done with `torch.cdist` + a 32-nearest-neighbour cap (matching `torch_geometric`'s default),
and `ase.data.atomic_masses` is a zeros buffer (dipole path unused) — so it runs in the `aevplig`
conda env with no extra installs.

`src/complex_graph.py` faithfully replicates GeoSSL's `TransformLBA` featurisation from our local
PDBbind structures: pocket = protein heavy atoms ≤ 6 Å of any ligand heavy atom (capped to
`maxnum` = 500 − n_ligand closest); ligand = ligand heavy atoms shifted +1000 Å along x
(`move_lig=True`, so the radius graph never connects pocket↔ligand); each atom → categorical index
into `[1,6,7,8,9,15,16,17,"unknown"]`, H dropped, out-of-vocab → "unknown".

`src/extract_features.py` encodes each **unique** complex once (features depend only on the
complex, not the split) → `features/schnet_feats.pt` (pid → 128-d vector). `src/probe.py` trains
the 2-layer MLP (128→256→1, dropout 0.2, standardized X and y, Adam 1e-3, best-val-RMSE early stop),
5 seeds (0–4), and writes `results/results_GEOSSL_<tag>.json` + `preds_GEOSSL_<tag>_seed*.jsonl`.

Structures: `dataset/data/pdbbind/structures/{pbpp-2020,misato_qm_built}/<pid>/`.
Labels (pK): `dataset/data/pdbbind/raw/LP_PDBBind.csv`. Feature coverage: 5986/5987
complexes across LP/CL/seq-id splits (1 build failure: `2r1w`); 5 holdout pids lack structures.

## Results — 5-seed mean ± SD (Pearson r / Spearman ρ / RMSE)

### LP-PDBBind (in-distribution; train & test within each tier)

| tier | n_test | Pearson r | Spearman ρ | RMSE |
|---|---|---|---|---|
| lp_edrscc_v2 (No-CL) | 1320 | 0.519 ± 0.012 | **0.499 ± 0.013** | 1.561 ± 0.024 |
| +CL1                 | 1166 | 0.525 ± 0.008 | 0.495 ± 0.004 | 1.570 ± 0.006 |
| +CL1+CL2             | 1149 | 0.535 ± 0.008 | 0.504 ± 0.007 | 1.538 ± 0.013 |
| +CL1+CL2+CL3         |  733 | 0.492 ± 0.020 | 0.455 ± 0.019 | 1.568 ± 0.016 |

### Sequence-identity splits on LP-PDBBind (ATOM3D-style; the requested test)

| split | n_test | Pearson r | Spearman ρ | RMSE |
|---|---|---|---|---|
| seq-id ≤ 60% (lba60) | 261 | 0.732 ± 0.004 | 0.720 ± 0.005 | 1.369 ± 0.013 |
| seq-id ≤ 30% (lba30) | 340 | 0.574 ± 0.011 | 0.528 ± 0.008 | 1.421 ± 0.031 |

### External test (probe trained on lp_edrscc_v2 train)

| set | n_test | Pearson r | Spearman ρ | RMSE |
|---|---|---|---|---|
| CASF-2016 core (leaky) | 214 | 0.718 ± 0.025 | 0.700 ± 0.018 | 1.501 ± 0.048 |
| CASF-2016 non-train    | 124 | 0.612 ± 0.017 | 0.601 ± 0.009 | 1.634 ± 0.031 |
| CASF-2016 clean-92     |  92 | 0.592 ± 0.019 | 0.585 ± 0.010 | 1.656 ± 0.031 |
| holdout2019 (all non-train) | 699 | 0.503 ± 0.009 | 0.449 ± 0.013 | 1.689 ± 0.021 |

**Takeaways.** On the canonical lp_edrscc_v2 split GeoSSL's frozen probe (ρ 0.499) is the
**weakest structural baseline** (vs C+D+G 0.644, ProFSA 0.597, GET 0.591, AEV-PLIG 0.550) — a
0.6 M SchNet pretrained on *single small molecules* transfers only weakly to protein–ligand
complexes when frozen. Correlation is much higher on the seq-id-60 split (ρ 0.72, redundant test)
and on the curated CASF-2016 core, and drops on the hard seq-id-30 / holdout / CL3-novel sets — the
expected difficulty ordering. GeoSSL-DDM was pretrained only on small molecules (no protein
structures), so there is **no test leakage** through pretraining.

## Reproduce

```bash
PY="conda run --no-capture-output -n aevplig python"
cd base/geossl
# features (encode each complex once across all target splits)
$PY src/extract_features.py --splits atom3d_lba60_edrscc_v2 atom3d_lba30_edrscc_v2 \
    lp_edrscc_v2 lp_edrscc_v2_cl1 lp_edrscc_v2_cl12 lp_edrscc_v2_cl123 \
    casf2016_eval casf2016_clean holdout2019_eval --device cuda:3
# in-distribution + seq-id probes
$PY src/probe.py --split atom3d_lba60_edrscc_v2 --tag lba60
$PY src/probe.py --split atom3d_lba30_edrscc_v2 --tag lba30
for t in v2:lp_edrscc_v2 cl1:lp_edrscc_v2_cl1 cl12:lp_edrscc_v2_cl12 cl123:lp_edrscc_v2_cl123; do
  $PY src/probe.py --split ${t#*:} --tag ${t%:*}; done
# external tests (train on lp_edrscc_v2)
$PY src/probe.py --split lp_edrscc_v2 --test_from casf2016_eval  --tag casf_core
$PY src/probe.py --split lp_edrscc_v2 --test_from casf2016_eval  --nontrain_only --tag casf_nontrain
$PY src/probe.py --split lp_edrscc_v2 --test_from casf2016_clean --tag casf_clean92
$PY src/probe.py --split lp_edrscc_v2 --test_from holdout2019_eval --tag holdout2019
```

Its scores (Table 1, CASF clean-92) are written to `base/_casf/GeoSSL.json`.

## Checkpoint availability (the other requested baseline)

DeepAffinity (Karimi et al., *Bioinformatics* 2019) also ships pretrained checkpoints, but they are
TensorFlow 1.1 + TFLearn 0.3 and require the SPS/SCRATCH sequence pipeline — not runnable in a
probe-comparable form. It is therefore listed as an excluded baseline rather than run.
