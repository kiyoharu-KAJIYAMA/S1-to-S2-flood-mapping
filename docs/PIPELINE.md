# Pipeline, step by step

Every path below is relative to the workspace root (`/home/kk/jp_claude/RMSD/` locally, `$RMSD_ROOT` on the
server). `P` means `processed/`. Scripts marked (rasterio) ran with `/usr/bin/python3`; scripts marked (torch)
ran with a CUDA PyTorch environment. This page is the chain that produced the paper; the reasons behind the
pairing, masking and split decisions are in `DATA_PREPARATION.md`.

## 1. Download

| Step | Script | Output |
|---|---|---|
| Sen1Floods11 hand-labeled chips and splits | `scripts/01_download/download_sen1floods11.py` | `data/Sen1Floods11/v1.1/` |
| C2S-MS Floods chips | `scripts/01_download/download_c2smsfloods.py` | `data/C2SMSFloods/` |
| FloodPlanet | `scripts/01_download/download_floodplanet.sh` | `data/FloodPlanet/` |
| STURM-Flood | `scripts/01_download/download_sturmflood.sh` | `data/STURMFlood/` |
| Kuro Siwo GRD archives | `scripts/01_download/download_kurosiwo.sh`, `stream_kurosiwo.sh` with `scripts/02_preprocess/harvest_kurosiwo.py` | `data/KuroSiwo/catalogue.gpkg`, `data/KuroSiwo_subset/` (flood tiles only, the 580 GB raw batches were deleted after extraction) |
| SEN12-FLOOD | manual download from IEEE DataPort | `data/SEN12FLOOD/trn/`, `tst/` |

Google Earth Engine exports (project `rmsd-503302`, Drive folder `Google Earth Engine`; check access with
`gee_check_auth.py`):

| Step | Script | Output |
|---|---|---|
| Kuro Siwo export rectangles around flooded tiles of the training events | `scripts/02_preprocess/kuro_flood_clusters.py` | `P/kuro_v9_export_regions.csv` |
| Kuro Siwo S1/S2 pairs, first pass (S1 within +-18 h of the flood time, single orbit; S2 least cloudy day, s2cloudless > 40 masked) | `scripts/01_download/gee_export_kuro_v9.py --submit` | `KURO_<region>_S1/S2.tif` |
| Kuro Siwo second pass with a wider window for the events the first pass missed | `scripts/01_download/gee_kuro_recover.py`, then `gee_export_kuro_recover.py --submit` | `KURR_<region>_S1/S2.tif` |
| STURM-Flood AOIs, same recipe, with JRC masks | `scripts/01_download/gee_export_sturm_v14.py --submit` (recipe of `gee_export_sturm_aois.py`) | `STURM_<region>_S1/S2.tif`, `STURMJRC_<region>.tif` |
| JRC masks for the other components | `gee_export_jrc.py` (KURR), `gee_export_kuro_jrc.py` (KURO, same grid as the S2 export), `gee_export_jrc_base.py` (the shipped-pair datasets) | `<PRE>JRC_<region>.tif`, `BASEJRC_<tile>.tif` |

## 2. Preprocess

Conventions shared by every component: 512 x 512 chips at 10 m, X = [VV, VH, RVI] in linear power
(RVI = 4 VH / (VV + VH)), Y = [MNDWI(B3, B11), NDVI(B8, B4), NDWI(B3, B8)], cloud and invalid pixels NaN in Y,
JRC occurrence >= 80% NaN in Y, NaN padding where a chip is smaller than 512.

| Step | Script (rasterio) | Output |
|---|---|---|
| SEN12-FLOOD scene pairs (nearest date, <= 1 day on flood dates, pixel cloud mask, scene rejection at > 50% cloud) | `scripts/02_preprocess/RMSD_preprocessing_v2.py` | `P/trn2_*` |
| Sen1Floods11 chips, labels and JRC layer in pipeline form | `scripts/02_preprocess/prep_s1f11.py` | `P/s1f11_X_VV_VH_RVI_raw.npy`, `s1f11_label.npy`, `s1f11_jrc.npy`, `s1f11_chips.csv` |
| Base pool: SEN12-FLOOD + Sen1Floods11 train/valid + C2S-MS (chips crossing the Sen1Floods11 test bboxes removed) | `build_trn4.py` | `P/trn4_*` |
| + FloodPlanet pairs | `build_trn5.py` | `P/trn5_*` |
| Flood-date pairs only | `build_trn6.py` | `P/trn6_*` (1,500 chips) |
| Georeference of every trn6 chip, then JRC applied to its target | `base_chip_geo.py`, `build_base_jrc.py` | `P/trn6_geo_index.csv`, `P/_basejrc_*` |
| Kuro Siwo MLU labels on the export grid (only used for chip acceptance) | `kuro_mlu_to_grid.py` | `data/KuroSiwo/mlu_on_grid/` |
| Kuro Siwo chips from the first-pass export (accept: valid >= 80%, cloud <= 10%, water >= 0.5%, cover >= 35%) | `build_trn9_chips.py`, JRC added by `build_trn9_jrc.py` | `P/_trn9new_*`, `P/_trn9jrc_*` |
| Kuro Siwo second-pass and STURM-Flood chips with JRC | `build_extra_chips_v14.py KURR _kurr2new`, `build_extra_chips_v14.py STURM _sturmnew` (base: `build_extra_chips.py`) | `P/_kurr2new_*`, `P/_sturmnew_*` |
| Concatenate the components | `build_trn16.py` | `P/trn16_X_VV_VH_RVI_raw.npy`, `trn16_Y_MNDWI_NDVI_NDWI_raw.npy`, `trn16_matched_samples_good.csv` (4,875 rows, GFD included) |
| Component and scene index for the trainer | `make_component_index.py` | `P/trn16_component.csv` |
| Chip metadata: dates, bboxes, source ids | `scripts/06_release/enrich_chip_meta.py` | `P/trn16_chip_meta.csv` |
| Test events held out, same floods excluded from training | `make_split_assignment.py` | `P/trn16_split.csv` |
| Flood clusters across datasets (50 km, 15 days) and the chip-level pool with every fold, including `fold_Paper` | `scripts/04_eval/loco_folds.py` (uses `flood_event_clusters.py`) | `experiments/loco_pool.csv` |
| Observed Sentinel-2 indices and validity of the Sen1Floods11 chips | `build_axis_observed_optical.py` | `P/s1f11_Yobs_MNDWI_NDVI_NDWI.npy`, `s1f11_Yobs_valid.npy` |
| Paper pool on the server: trn16 rows of `fold_Paper == train` plus the 48 official-test Sen1Floods11 chips of training events | `terra/08_loco/build_trn18.py` | `output/loco/root_Paper/processed/trn18_*` |

The trainer applies the last two filters at load time: target water share >= 0.5% (`--drop_low_water 0.005`,
2,844 chips left) and at least 20% valid Sentinel-1 and target pixels (2,735 left, Table I).

## 3. Train (server, torch)

```
python terra/12_l1mndwi0/make_configs.py --tier 0        # writes configs/P0-s1..5.json from the loss-ablation template
bash terra/12_l1mndwi0/worker_onedir.sh <gpu> <cpuset> terra/12_l1mndwi0/queue_tier0.txt
```

`worker_onedir.sh` passes every key of a config as a command-line flag to `scripts/09_onedir/train_onedir.py`
(one direction, SAR to MSI; the two-direction equivalent is `scripts/06_ablation/train_variant.py`, itself the
`scripts/03_train/RMSD_model_bidirectional_fast2.py` pipeline with input and architecture hooks). The settings
that matter are in `terra/12_l1mndwi0/configs/P0-s1.json`: `loss L1`, `mndwi_weight 1.0`, `kl_weight 0`,
`dice_weight 0`, `optimizer AdamW`, `lr 1e-4` (`lr_factor 0.5`, `lr_patience 5`), `bs 16`, `crop 256`,
`augment true`, `total_steps 14000`, `val_frac 0.2` (scene level), `amp true`, `data_prefix trn18`,
`components sen12,s1f11,c2s,fp,kuros2,kurr,sturm`, `split_keep train`, `drop_low_water 0.005`, `arch unet`,
`in_mask 1,1,1`. Seeds 1 to 5 differ only in `seed` and `date_out`. Backscatter is converted to dB and normalized
per channel inside the trainer (statistics of the training scenes, stored in the checkpoint); no speckle filter.
Each run writes `best_model.pt`, `config_*.json`, `scene_split.json` and `loss_history.csv` (`scene_split.json` is in
`runs/`).

## 4. Postprocess (server, torch)

```
python terra/12_l1mndwi0/eval_tier0.py
```

For each seed the checkpoint translates the 139 test chips, translated MNDWI is un-normalized with the checkpoint
`Y_min`/`Y_max` and cut at 0, and the five seeds vote (water when >= 3). `eval_tier0.py` writes
`output/l1mndwi0/metrics_tier0.json` (Table II and the Results numbers, copy in `results/`), the packed masks
`masks_P0_A.npy` and `masks_P0-s<k>_A.npy`, and the five-seed mean translated MNDWI. It imports the loaders and
scorers of `scripts/06_ablation/common.py`, `terra/10_sameday/eval_sameday.py` and `eval_loss_ablation.py`.

Locally:

```
python scripts/04_eval/votes_from_terra_masks.py output/from_terra/l1mndwi0_20261026/masks_P0_A.npy
```

writes `output/water_states/testA_translation_votes.npy`, the vote file of every figure. `--check` compares
instead of writing (0 differing pixels against the file used for the paper).

## 5. Analysis and figures (local)

Caches built once:

| Cache | Script | Output |
|---|---|---|
| Lee-filtered (7 x 7, ENL 4.4) VV and VH in dB of the Sen1Floods11 chips | `scripts/04_eval/eval_optical_axis.py --axis s1f11 --run <any run>` (function `lee`) | `P/s1f11_lee_db.npz` |
| WorldCover 2020 on the 139 test chips | `scripts/04_eval/testA_landcover.py` | `P/s1f11_testA_worldcover_2020.npy` |

Paper items (all read the vote file through `scripts/05_figures/testA_common.py`):

```
cd scripts/05_figures
RMSD_MODELS=P0 python fig_architecture_testA.py                 # Fig. 1 (torch; P0 checkpoints in output/from_terra/ckpt_P0/)
python fig_overall1_testA.py                                   # Fig. 2, chip India_747992
python fig_recovery_distance_testA.py                          # Fig. 3, crop India_399883 at (209, 185)
python fig_unrecovered_space_testA.py                          # Fig. 4, window of India_747992
python fig_recovery_state_testA.py --scale 0.9                 # Fig. 5
python table3_landcover_shares.py                              # Table III
python ../04_eval/table1_pool_events.py                        # Table I
python ../04_eval/testA_results_numbers.py                     # every Results number from the vote file
```

Outputs go to `output/water_states/fig_*_testA.{pdf,png}`; the PDFs were renamed `fig_architecture.pdf`,
`fig_overall.pdf`, `fig_recovery_distance.pdf`, `fig_unrecovered_space.pdf` and `fig_recovery_state.pdf` in the
manuscript.

Threshold calibration: the rule VV < -5 dB and VH < -22 dB is the F1-optimal pair on the Sen1Floods11 training
events, first on the 307 hand-labeled chips (`scripts/07_local_seg/threshold_rules.py`) and then re-fitted on the
243 chips the paper models trained on (`terra/08_loco/eval_paper.py`, rule "b"), which gives the same bounds.
