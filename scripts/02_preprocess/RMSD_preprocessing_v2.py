#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RMSD preprocessing v2 — revision that increases the number of pairs.

Changes from v1 (RMSD_preprocessing.py):
 1. S1/S2 matching: same-day only -> nearest-neighbour match
      flood day (FLOODING=True on either the S2 or the S1 side) : |day_diff| <= 1
      non-flood day                                             : |day_diff| <= 3
    (water extent changes drastically within days during a flood, so flood days are restricted tightly)
 2. Cloud: whole-scene rejection (cloud_fraction>0.01) -> per-pixel NaN in Y.
    The training loss is masked, so NaN pixels are ignored automatically.
    Scene rejection only for cloud_fraction > 0.5 or valid_fraction < 0.5.
 3. Output file names use the trn2_/tst2_ prefix (v1 npy files are kept unchanged).
    CSV gains s1_date, day_diff and flood flag columns.

X: [VV, VH, RVI] / Y: [MNDWI, NDVI, NDWI], 512x512, S2 R-band grid reference identical to v1.
Run: WSL native python (needs rasterio; not in the torch container).
"""
import os
import json
import glob as g
import numpy as np
import pandas as pd
import rasterio as rio
from datetime import datetime
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kw):
        """Minimal progress printer for environments without tqdm."""
        items = list(iterable)
        n = len(items)
        step = max(1, n // 20)
        for i, item in enumerate(items):
            if i % step == 0 or i == n - 1:
                print(f"{desc or 'progress'}: {i + 1}/{n}", flush=True)
            yield item
from rasterio.warp import reproject, Resampling

SEN12FLOOD_root = "/home/kk/jp_claude/RMSD/data/SEN12FLOOD"
out_dir = os.path.join(SEN12FLOOD_root, "processed")

WINDOW_FLOOD = 1      # allowed day gap on flood days
WINDOW_NONFLOOD = 3   # allowed day gap on non-flood days
MAX_CLOUD_FRACTION = 0.5
MIN_VALID_FRACTION = 0.5

# ------------------------------------------------------------------
# helpers identical to v1
# ------------------------------------------------------------------
def parse_s1_date(path):
    name = os.path.basename(path)
    return datetime.strptime(name.split('_')[5].split('T')[0], '%Y%m%d')

def parse_s2_date(path):
    name = os.path.basename(path)
    return datetime.strptime(name.split('_')[1], '%Y-%m-%d')

def read_tif(path):
    with rio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr <= -9990] = np.nan
    return arr

def read_reproject_to_match(src_path, match_path, resampling=Resampling.bilinear):
    with rio.open(match_path) as match:
        dst_crs = match.crs
        dst_transform = match.transform
        dst_height = match.height
        dst_width = match.width
    with rio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan
        src_arr[src_arr <= -9990] = np.nan
        dst_arr = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst_arr,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst_arr

def norm_diff(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / (a + b)
    out[~np.isfinite(out)] = np.nan
    return out

def cloud_like_mask(R, G, B, Nir,
                    brightness_threshold=0.28,
                    whiteness_threshold=0.16,
                    ndvi_threshold=0.35):
    """Pixel-mask version of v1 estimate_cloud_fraction with identical conditions."""
    valid = np.isfinite(R) & np.isfinite(G) & np.isfinite(B) & np.isfinite(Nir)
    visible_mean = (R + G + B) / 3.0
    visible_range = np.maximum.reduce([R, G, B]) - np.minimum.reduce([R, G, B])
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (Nir - R) / (Nir + R)
    cloud = (
        valid
        & (visible_mean > brightness_threshold)
        & (visible_range < whiteness_threshold)
        & (ndvi < ndvi_threshold)
    )
    return cloud, valid

# ------------------------------------------------------------------
# flood flags (S1list.json / S2list.json)
# ------------------------------------------------------------------
def load_flood_flags():
    """folder -> {'s1': {date: flood}, 's2': {date: flood}}"""
    flags = {}
    for key, fname in [("s1", "S1list.json"), ("s2", "S2list.json")]:
        with open(os.path.join(SEN12FLOOD_root, fname)) as f:
            data = json.load(f)
        for folder, info in data.items():
            if not isinstance(info, dict):
                continue
            d = flags.setdefault(folder, {"s1": {}, "s2": {}})
            for k, v in info.items():
                if isinstance(v, dict) and "date" in v:
                    try:
                        dt = datetime.strptime(v["date"], "%Y-%m-%d")
                    except ValueError:
                        continue
                    d[key][dt] = bool(v.get("FLOODING", False))
    return flags

FLOOD_FLAGS = load_flood_flags()

def flood_of(folder, sensor, date):
    return FLOOD_FLAGS.get(os.path.basename(folder), {}).get(sensor, {}).get(date, False)

# ------------------------------------------------------------------
# v2 matching: nearest S1 per S2 (flood day +/-1, non-flood day +/-3)
# ------------------------------------------------------------------
def build_matched_df_one_folder_v2(folder):
    # Note: the v1 pattern '*B04*.tif' also matched hex IDs in S1 file names (e.g. _01B043_)
    # and made whole folders fail. Tightened with the S2_ prefix.
    VVs = sorted(g.glob(os.path.join(folder, 'S1*_VV.tif')))
    VHs = sorted(g.glob(os.path.join(folder, 'S1*_VH.tif')))
    Rs = sorted(g.glob(os.path.join(folder, 'S2_*_B04.tif')))
    Gs = sorted(g.glob(os.path.join(folder, 'S2_*_B03.tif')))
    Bs = sorted(g.glob(os.path.join(folder, 'S2_*_B02.tif')))
    Nirs = sorted(g.glob(os.path.join(folder, 'S2_*_B08.tif')))
    B11s = sorted(g.glob(os.path.join(folder, 'S2_*_B11.tif')))

    df_vv = pd.DataFrame({'VV': VVs, 'date': [parse_s1_date(f) for f in VVs]})
    df_vh = pd.DataFrame({'VH': VHs, 'date': [parse_s1_date(f) for f in VHs]})
    df_s1 = df_vv.merge(df_vh, on='date', how='inner').drop_duplicates('date')

    df_s2 = None
    for col, files in [('R', Rs), ('G', Gs), ('B', Bs), ('Nir', Nirs), ('B11', B11s)]:
        d = pd.DataFrame({col: files, 'date': [parse_s2_date(f) for f in files]})
        df_s2 = d if df_s2 is None else df_s2.merge(d, on='date', how='inner')

    rows = []
    if df_s1.empty or df_s2 is None or df_s2.empty:
        return pd.DataFrame(rows)

    s1_dates = df_s1['date'].tolist()
    for _, r2 in df_s2.iterrows():
        d2 = r2['date']
        # nearest S1
        diffs = [(abs((d2 - d1).days), d1) for d1 in s1_dates]
        diffs.sort()
        best_diff, best_d1 = diffs[0]
        flood = flood_of(folder, 's2', d2) or flood_of(folder, 's1', best_d1)
        window = WINDOW_FLOOD if flood else WINDOW_NONFLOOD
        if best_diff > window:
            continue
        r1 = df_s1[df_s1['date'] == best_d1].iloc[0]
        rows.append({
            'VV': r1['VV'], 'VH': r1['VH'],
            'R': r2['R'], 'G': r2['G'], 'B': r2['B'],
            'Nir': r2['Nir'], 'B11': r2['B11'],
            'date': d2, 's1_date': best_d1,
            'day_diff': int(best_diff), 'flood': bool(flood),
            'folder': folder,
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# X/Y generation (same reprojection/index computation as v1 + per-pixel cloud NaN)
# ------------------------------------------------------------------
def make_xy_v2(df_matched, expected_shape=(512, 512)):
    X_list, Y_list = [], []
    good_records, bad_rows = [], []

    for i in tqdm(range(len(df_matched)), desc="making X/Y v2"):
        row = df_matched.iloc[i]
        try:
            match_path = row["R"]
            VV = read_reproject_to_match(row["VV"], match_path, Resampling.bilinear)
            VH = read_reproject_to_match(row["VH"], match_path, Resampling.bilinear)
            VV[VV < 0] = np.nan
            VH[VH < 0] = np.nan

            B = read_reproject_to_match(row["B"], match_path, Resampling.bilinear) / 10000.0
            G = read_reproject_to_match(row["G"], match_path, Resampling.bilinear) / 10000.0
            R = read_tif(row["R"]) / 10000.0
            Nir = read_reproject_to_match(row["Nir"], match_path, Resampling.bilinear) / 10000.0
            B11 = read_reproject_to_match(row["B11"], match_path, Resampling.bilinear) / 10000.0

            shapes = [VV.shape, VH.shape, B.shape, G.shape, R.shape, Nir.shape, B11.shape]
            if len(set(shapes)) != 1:
                bad_rows.append((i, row["folder"], row["date"], "shape mismatch", str(shapes)))
                continue
            if expected_shape is not None and VV.shape != expected_shape:
                bad_rows.append((i, row["folder"], row["date"], "not expected shape", str(shapes)))
                continue

            # per-pixel cloud mask; whole-scene rejection only in extreme cases
            cloud, valid = cloud_like_mask(R, G, B, Nir)
            valid_fraction = float(np.mean(valid))
            cloud_fraction = float(np.sum(cloud) / max(np.sum(valid), 1))
            if valid_fraction < MIN_VALID_FRACTION:
                bad_rows.append((i, row["folder"], row["date"], "low optical valid fraction",
                                 f"valid={valid_fraction:.3f}"))
                continue
            if cloud_fraction > MAX_CLOUD_FRACTION:
                bad_rows.append((i, row["folder"], row["date"], "cloudy optical scene",
                                 f"cloud={cloud_fraction:.3f}"))
                continue

            # RVI (same automatic dB detection as v1)
            if np.nanmedian(VV) < 0 and np.nanmedian(VH) < 0:
                VV_for_rvi = 10 ** (VV / 10.0)
                VH_for_rvi = 10 ** (VH / 10.0)
            else:
                VV_for_rvi = VV
                VH_for_rvi = VH
            with np.errstate(invalid="ignore", divide="ignore"):
                RVI = (4.0 * VH_for_rvi) / (VV_for_rvi + VH_for_rvi)
            RVI[~np.isfinite(RVI)] = np.nan

            MNDWI = norm_diff(G, B11)
            NDVI = norm_diff(Nir, R)
            NDWI = norm_diff(G, Nir)

            # set optical indices to NaN on cloud pixels (masked loss ignores them automatically)
            for arr in (MNDWI, NDVI, NDWI):
                arr[cloud] = np.nan

            X = np.stack([VV, VH, RVI], axis=0).astype(np.float32)
            Y = np.stack([MNDWI, NDVI, NDWI], axis=0).astype(np.float32)
            X_list.append(X)
            Y_list.append(Y)
            good_records.append({
                "idx": i,
                "cloud_fraction": cloud_fraction,
                "optical_valid_fraction": valid_fraction,
            })
        except Exception as e:
            bad_rows.append((i, row.get("folder"), row.get("date"), str(e), None))

    if len(X_list) == 0:
        X_arr = np.empty((0, 3, *expected_shape), dtype=np.float32)
        Y_arr = np.empty((0, 3, *expected_shape), dtype=np.float32)
        df_good = df_matched.iloc[[]].reset_index(drop=True)
    else:
        X_arr = np.stack(X_list, axis=0)
        Y_arr = np.stack(Y_list, axis=0)
        info = pd.DataFrame(good_records)
        df_good = df_matched.iloc[info["idx"].to_numpy()].reset_index(drop=True)
        df_good["cloud_fraction"] = info["cloud_fraction"].to_numpy()
        df_good["optical_valid_fraction"] = info["optical_valid_fraction"].to_numpy()

    df_bad = pd.DataFrame(bad_rows, columns=["idx", "folder", "date", "reason", "detail"])
    return X_arr, Y_arr, df_good, df_bad

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def process_split(split_name, folders):
    print("=" * 70)
    print(f"Processing split: {split_name}  (v2: flood +/-{WINDOW_FLOOD}d, nonflood +/-{WINDOW_NONFLOOD}d)")
    print(f"N folders: {len(folders)}")
    print("=" * 70, flush=True)

    df_list = []
    for folder in tqdm(folders, desc="matching v2"):
        try:
            d = build_matched_df_one_folder_v2(folder)
            if len(d) > 0:
                df_list.append(d)
        except Exception as e:
            print(folder, "failed:", e)
    df_matched = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
    print(f"{split_name} matched pairs: {len(df_matched)} "
          f"(flood={int(df_matched['flood'].sum()) if len(df_matched) else 0})", flush=True)
    if len(df_matched) == 0:
        return

    df_matched.to_csv(os.path.join(out_dir, f"{split_name}2_matched_samples_all.csv"), index=False)

    X_arr, Y_arr, df_good, df_bad = make_xy_v2(df_matched)
    print(f"{split_name} X: {X_arr.shape}  good={len(df_good)} bad={len(df_bad)}")
    if len(df_bad) > 0:
        print(df_bad["reason"].value_counts())

    np.save(os.path.join(out_dir, f"{split_name}2_X_VV_VH_RVI_raw.npy"), X_arr)
    np.save(os.path.join(out_dir, f"{split_name}2_Y_MNDWI_NDVI_NDWI_raw.npy"), Y_arr)
    df_good.to_csv(os.path.join(out_dir, f"{split_name}2_matched_samples_good.csv"), index=False)
    df_bad.to_csv(os.path.join(out_dir, f"{split_name}2_matched_samples_bad.csv"), index=False)
    print(f"saved {split_name}2_* to {out_dir}", flush=True)

if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    trns = sorted([f for f in g.glob(os.path.join(SEN12FLOOD_root, "trn", "*")) if os.path.isdir(f)])
    tsts = sorted([f for f in g.glob(os.path.join(SEN12FLOOD_root, "tst", "*")) if os.path.isdir(f)])
    process_split("trn", trns)
    process_split("tst", tsts)
    print("done.")
