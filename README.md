# What Sentinel-1-to-Sentinel-2 Translation Recovers and Leaves of the Flood Water Missed by Backscatter Thresholding

Code of the manuscript submitted to IEEE Geoscience and Remote Sensing Letters (2026).

A U-Net (S1toS2) translates Sentinel-1 backscatter (VV, VH, RVI) into the Sentinel-2 indices MNDWI, NDVI and NDWI.
It is trained on 2,735 Sentinel-1/Sentinel-2 pairs fused from six public flood datasets, with no manual labels, and
tested on 139 hand-labeled Sen1Floods11 chips of four held-out floods (Bolivia, India, Somalia, Spain) against a
backscatter threshold calibrated on the Sen1Floods11 training events (VV < -5 dB and VH < -22 dB, "VH-22").
The paper splits the labeled flood water into what VH-22 finds, what S1toS2 recovers and what remains, by distance
from detected water, by backscatter brightness and by ESA WorldCover land cover.

This repository holds the scripts that produced every number, table and figure of the paper, organized by stage.
The scripts are published as they were run. They are research code with hard-coded workspace paths (see
"Paths" below), not a package.

## Layout

```
scripts/01_download/   dataset downloads and Google Earth Engine exports (Kuro Siwo and STURM-Flood pairs, JRC masks)
scripts/02_preprocess/ chip building (512 x 512 at 10 m), cloud and permanent-water masking, pool assembly, splits
scripts/03_train/      the base trainer (RMSD_model_bidirectional_fast2.py)
scripts/06_ablation/   the trainer with input/architecture hooks used for every server run (train_variant.py),
                       and the shared Test A loader and scorer (common.py)
scripts/04_eval/       U-Net definition, Lee-filter cache, flood clustering, pool definition, Test A numbers, Table I
scripts/05_figures/    the five figures and Table III
scripts/07_local_seg/  why the threshold is VV < -5 and VH < -22 dB (F1 calibration on the training events)
terra/                 scripts that ran on the GPU server: pool trn18, training queue, P0 configs, Test A evaluation
runs/P0-s1..5/         scene-level train/validation split of the five paper models (checkpoints are not on GitHub, 69 MB each)
results/               metrics_tier0.json, the server evaluation of P0 that Table II and the Results text quote
docs/                  PIPELINE.md (every step with its command), DATA_PREPARATION.md (pairs, masks, leak removal,
                       splits), MANIFEST.md (where each file lived in the workspace)
```

No data is redistributed here: no imagery, chips, labels, masks, derived arrays or checkpoints. The scripts in
`scripts/01_download/` fetch every dataset from its original source (see "Data"), and the rest of the pipeline
rebuilds everything from those downloads.

## From the paper to the scripts

| Paper item | Script | Input it reads |
|---|---|---|
| Fig. 1 (training and test scheme) | `scripts/05_figures/fig_architecture_testA.py` (`RMSD_MODELS=P0`) | P0 checkpoints, chip India_747992 |
| Fig. 2 (one chip, four classes) | `scripts/05_figures/fig_overall1_testA.py` | vote file, Sen1Floods11 S2Hand |
| Fig. 3 (recovery by distance) | `scripts/05_figures/fig_recovery_distance_testA.py` | vote file |
| Fig. 4 (unrecovered water in SAR and optical space) | `scripts/05_figures/fig_unrecovered_space_testA.py` | vote file |
| Fig. 5 (recovery against VH by distance bin) | `scripts/05_figures/fig_recovery_state_testA.py --scale 0.9` | vote file |
| Table I (training pool) | `scripts/04_eval/table1_pool_events.py` | `experiments/loco_pool.csv` (written by `loco_folds.py`), trn16 metadata |
| Table II (scores) | `terra/12_l1mndwi0/eval_tier0.py` on the server, `results/metrics_tier0.json`; locally `scripts/04_eval/testA_results_numbers.py` | P0 checkpoints / vote file |
| Table III (land cover) | `scripts/05_figures/table3_landcover_shares.py` | vote file, WorldCover cache |
| VH-22 calibration | `terra/08_loco/eval_paper.py` (243 chips, rule b), `scripts/07_local_seg/threshold_rules.py` | Sen1Floods11 training events |

The vote file is `output/water_states/testA_translation_votes.npy`, the five-seed majority (>= 3 of 5) of translated
MNDWI > 0 on the 139 test chips. It is unpacked from the server mask `masks_P0_A.npy` by
`scripts/04_eval/votes_from_terra_masks.py`. All figure scripts read it through `scripts/05_figures/testA_common.py`,
which also fixes the pixel set (labeled, valid Sentinel-2 observation, no permanent water, finite SAR), the
threshold, the distance transform and the true-colour display rule.

## Pipeline in one page

Full details and commands are in [docs/PIPELINE.md](docs/PIPELINE.md).

1. **Download** the six datasets (Sen1Floods11, C2S-MS Floods, FloodPlanet, SEN12-FLOOD, Kuro Siwo, STURM-Flood).
   Kuro Siwo and STURM-Flood ship no usable Sentinel-2, so their pairs are exported from Google Earth Engine
   (`gee_export_kuro_v9.py`, `gee_export_kuro_recover.py`, `gee_export_sturm_v14.py`), together with the JRC Global
   Surface Water occurrence >= 80% masks (`gee_export_jrc*.py`).
2. **Preprocess** every dataset into 512 x 512 chips at 10 m with X = [VV, VH, RVI] in linear power and
   Y = [MNDWI, NDVI, NDWI], cloud pixels and permanent water set to NaN in Y
   (`RMSD_preprocessing_v2.py`, `prep_s1f11.py`, `build_trn4/5/6.py`, `build_base_jrc.py`, `build_trn9_chips.py`,
   `build_extra_chips_v14.py`, `build_trn9_jrc.py`). Concatenate into the trn16 array (`build_trn16.py`), index its
   components (`make_component_index.py`, `enrich_chip_meta.py`), assign test events (`make_split_assignment.py`),
   cluster floods across datasets and write the chip-level pool with the paper split (`loco_folds.py`).
   On the server, `terra/08_loco/build_trn18.py` builds the paper pool trn18 (trn16 rows plus the 48 official-test
   Sen1Floods11 chips of training events). The trainer then drops chips with less than 0.5% water or less than 20%
   valid pixels, leaving 2,735.
3. **Train** five seeds of the paper model P0 with `terra/12_l1mndwi0/configs/P0-s*.json`
   (L1 loss only, three indices weighted equally, AdamW 1e-4 halved on plateau, batch 16, 256 x 256 crops with
   flip and rotation, 14,000 updates, scene-level 80/20 validation split, AMP). The configs are consumed by
   `terra/12_l1mndwi0/worker_onedir.sh`, which calls the one-direction trainer (`train_onedir.py`, see "Status").
4. **Postprocess**: each checkpoint translates the 139 test chips, a pixel is water when translated MNDWI > 0, and the
   five seeds vote (`terra/12_l1mndwi0/eval_tier0.py`, writing `masks_P0_A.npy` and `metrics_tier0.json`).
   Locally, `votes_from_terra_masks.py` unpacks the mask into the vote file.
5. **Analyse and draw**: `testA_landcover.py` samples WorldCover 2020 onto the test chips (cached),
   `eval_optical_axis.py --axis s1f11` caches the 7 x 7 Lee-filtered dB backscatter (`s1f11_lee_db.npz`),
   `build_axis_observed_optical.py` makes the observed Sentinel-2 indices and validity of the test chips, and the
   scripts of the table above produce the figures and tables.

## Data

| Dataset | Source | Used for |
|---|---|---|
| Sen1Floods11 v1.1 | GCS bucket `sen1floods11` (`download_sen1floods11.py`) | training (7 events), test (4 events), threshold calibration |
| C2S-MS Floods | AWS `radiant-mlhub/c2smsfloods` (`download_c2smsfloods.py`) | training |
| FloodPlanet | Zenodo 15238572 (`download_floodplanet.sh`) | training |
| SEN12-FLOOD | IEEE DataPort (manual download, `trn/` and `tst/` scene folders) | training |
| Kuro Siwo | Dropbox archives listed in `download_kurosiwo.sh`, Sentinel-2 from GEE | training |
| STURM-Flood | Zenodo 12748982 (`download_sturmflood.sh`), pairs re-exported from GEE | training |
| JRC Global Surface Water v1.4 | GEE `JRC/GSW1_4/GlobalSurfaceWater` | permanent-water mask |
| ESA WorldCover 2020 v100 | public COGs on AWS (`testA_landcover.py`) | Table III, Fig. 3 |

GFD and S1S2-Water chips were also built during the project (`build_extra_chips_v14.py GFD`, `build_trn10a.py`) but
are not part of the paper pool, which the trainer selects with `--components sen12,s1f11,c2s,fp,kuros2,kurr,sturm`.

## Paths

Local scripts hard-code the workspace root `/home/kk/jp_claude/RMSD/` with `data/`, `processed/` and `output/`
beneath it, and GEE exports are read from the Google Drive folder `/mnt/d/gdrive/Google Earth Engine/`.
Server scripts (`scripts/06_ablation/`, `terra/`) read the root from the environment variable `RMSD_ROOT`
(default `/lustre/home/kajiyama/rmsd`). Replace these roots before running.
Figure scripts accept `RMSD_VOTES` (another vote file) and `RMSD_O` (another output folder) so that a candidate
model never overwrites the paper figures.

## Environment

Local analysis and figures: Python 3.10, numpy 1.26, scipy 1.15, matplotlib 3.10, rasterio 1.4, pandas 2.2,
earthengine-api 1.7 (exports only). Training and server evaluation: PyTorch 2.x with CUDA. See `requirements.txt`.

## Status

- `terra/09_onedir/train_onedir.py`, the one-direction trainer that `worker_onedir.sh` calls, and
  `terra/11_loss_ablation/eval_loss_ablation.py`, imported by `eval_tier0.py` for the MNDWI = 0 cut, are still on
  the server and will be added. `train_onedir.py` is `scripts/06_ablation/train_variant.py` without the
  MSI-to-SAR branch.
- Checkpoints (`best_model.pt`, 69 MB per seed) and the chip arrays (23 GB) are not on GitHub.

## Citation

K. Kajiyama, J. Lee, E. Lee, H. Kim, H.-L. Choi, H. Kim and S. J. Kim, "What Sentinel-1-to-Sentinel-2 Translation
Recovers and Leaves of the Flood Water Missed by Backscatter Thresholding", submitted to IEEE Geoscience and Remote
Sensing Letters, 2026.
