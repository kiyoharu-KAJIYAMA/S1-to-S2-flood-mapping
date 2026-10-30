# Data preparation: what each component went through

English summary of the project's preprocessing audit (the audit was written against the code of the final pool,
not from memory; the script that implements each item is named).

## Seven sources, six in the paper

Seven public flood datasets were acquired: Kuro Siwo, GFD, STURM-Flood, Sen1Floods11, C2S-MS Floods, SEN12-FLOOD
and FloodPlanet. GFD chips were built but are not in the paper pool (the trainer selects components with
`--components sen12,s1f11,c2s,fp,kuros2,kurr,sturm`). S1S2-Water (65 tiles, all non-flood) and OMBRIA were
acquired and dropped. Kuro Siwo appears as two components, `kuros2` (first GEE export) and `kurr` (second export
with a wider window for the events the first pass missed); the events are disjoint and the paper counts them as
one dataset.

No flood label is used for training. The target is the observed Sentinel-2 indices. The only selection for the
pool is "can a near-simultaneous Sentinel-1/Sentinel-2 pair be made". Labels are used for the test and for
calibrating the backscatter rule only.

## Pairs

| Dataset | Pair source | Realized abs(dt) days, min / median / max |
|---|---|---|
| FloodPlanet | shipped | 0 / 0 / 0 |
| Sen1Floods11 | shipped | 0 / 0 / 2 |
| SEN12-FLOOD | nearest date in the shipped time series, <= 1 day on flood dates, <= 3 days otherwise | 0 / 1 / 1 |
| C2S-MS Floods | shipped | 0 / 1 / 4 |
| STURM-Flood | GEE: S1 within +-18 h of the flood time, single orbit mosaic; least cloudy S2 within -18/+42 h | 0 / 1 / 2 |
| Kuro Siwo | GEE, same recipe (second pass: per-pixel nearest clear S2 within the widened window) | 0 / 2 / 3 (second pass 1 / 1 / 6) |

Scripts: `RMSD_preprocessing_v2.py` (SEN12-FLOOD), `gee_export_kuro_v9.py`, `gee_export_kuro_recover.py`,
`gee_export_sturm_v14.py`. Measured dt per chip is in `trn16_chip_meta.csv` (`enrich_chip_meta.py`).

## Cloud and invalid pixels

Four mechanisms, all producing NaN in the target Y:

| Mechanism | Applied to |
|---|---|
| s2cloudless probability > 40 masked at export | the GEE components (Kuro Siwo, STURM-Flood, GFD) |
| brightness > 0.28 and whiteness < 0.16 and NDVI < 0.35 (`RMSD_preprocessing_v2.cloud_like_mask`) | the shipped chips (Sen1Floods11, FloodPlanet, C2S-MS) |
| shipped LabelCloud, OR-ed with the heuristic | C2S-MS only |
| scene rejection at cloud > 50% or valid < 50% | SEN12-FLOOD only |

The GEE components also pass a chip acceptance test: valid >= 80%, cloud <= 10%, MNDWI > 0 share >= 0.5%
(after the JRC mask), tile coverage >= 35% (`build_trn9_chips.py`, `build_extra_chips_v14.py`). The shipped chips
are already cut, so no acceptance test applies to them.

## Permanent water

JRC Global Surface Water v1.4 occurrence >= 80% is set to NaN in the target of every training chip
(`build_base_jrc.py`, `build_trn9_jrc.py`, `build_extra_chips_v14.py`) and excluded from the scored pixels of the
test chips (`testA_common.py`, `common.py`). The learning and scoring target is therefore water where there is
normally none.

## Leak removal and splits

The four test floods (Sen1Floods11 Bolivia, India, Somalia, Spain) are recorded by other datasets as well:
C2S-MS (72 chips: India 47, Somalia 19, Spain 6) and FloodPlanet (88 chips: Bolivia 63, Somalia 17, Spain 8).
These chips are excluded from training (`make_split_assignment.py`, column `split`; `loco_folds.py`, column
`fold_Paper`). Floods shared by two training-side datasets (for example FloodPlanet and Sen1Floods11 Ghana,
Nigeria, Paraguay) stay in the pool. C2S-MS chips whose bbox crosses a Sen1Floods11 test chip were dropped at
chip-building time (`build_trn4.py`). Floods are linked across datasets when their centres are within 50 km and
their Sentinel-1 dates within 15 days (`flood_event_clusters.py`); Table I counts each linked flood once.

The 64 official-test Sen1Floods11 chips that belong to the seven training events were added to the pool for the
paper model (`terra/08_loco/build_trn18.py`): 48 remain after the water-share and validity filters, giving
Sen1Floods11 243 chips.

The validation split is at the scene level, not the event level: the trainer shuffles scenes with the run seed and
holds out 20% (`val_frac 0.2`), and a GEE event can span up to 13 scenes. Validation only selects the checkpoint;
every reported number comes from the event-level held-out test chips.

## Filters applied by the trainer

Chips whose target has less than 0.5% water are dropped (`--drop_low_water 0.005`), leaving 2,844; chips with
fewer than 20% valid Sentinel-1 pixels (finite and positive VV) or fewer than 20% finite target MNDWI pixels are
dropped, leaving 2,735 (Table I). Normalization statistics are computed on the training scenes only and stored in
the checkpoint.

## Known asymmetries

1. The cloud mask is not one method but four, following each dataset's distribution format.
2. The chip acceptance test applies only to the three GEE components.
3. FloodPlanet and SEN12-FLOOD provide Level-2A reflectance, the others Level-1C.
4. Kuro Siwo MLU labels and the FloodPlanet labels were not used for calibration or scoring in this paper.
