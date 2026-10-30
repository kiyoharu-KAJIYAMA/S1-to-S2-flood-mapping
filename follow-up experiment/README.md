# Server-side scripts

These ran on the GPU server under `$RMSD_ROOT/scripts/<dir>/` and are copied from the take-home archives of each
run. They import each other by directory name, so keep the directory names.

| Directory | Role in the paper |
|---|---|
| `08_loco/` | `build_trn18.py` builds the paper pool trn18 from `loco_pool.csv`; `eval_paper.py` calibrates the backscatter rule and the earlier model on the 243 training chips; `eval_loco.py` holds the calibration and scoring functions |
| `10_sameday/` | `eval_sameday.py` holds the Table II scorer (`table2`) that `12_l1mndwi0/eval_tier0.py` reuses; `build_pools.py` and `build_pool_ms.py` build the same-day pools of the reply to the co-authors (Tier 2) |
| `12_l1mndwi0/` | the paper model P0: `make_configs.py` (configs), `worker_onedir.sh` (GPU queue), `eval_tier0.py` (Table II, masks, mean translated MNDWI), `eval_tier1.py` and `cell_thresholds.py` (input ablation of the reply), `eval_tier2.py` (same-day pools) |

Not yet copied from the server (ssh was unavailable when this folder was built):

- `09_onedir/train_onedir.py`, the trainer that `12_l1mndwi0/worker_onedir.sh` calls
  (`scripts/06_ablation/train_variant.py` without the MSI-to-SAR branch)
- `11_loss_ablation/eval_loss_ablation.py`, imported by `eval_tier0.py` (`tau_mndwi0`, `WC_FILE`)
