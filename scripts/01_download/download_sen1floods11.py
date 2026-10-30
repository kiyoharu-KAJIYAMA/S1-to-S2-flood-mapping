#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download the hand-labeled part of Sen1Floods11 (Bonafilia et al. 2020).
Fetched anonymously over HTTPS from the public GCS bucket sen1floods11 (no gsutil needed).

Targets (v1.1):
  data/flood_events/HandLabeled/{S1Hand, S2Hand, LabelHand, JRCWaterHand}
  splits/flood_handlabeled/  (train/valid/test split CSVs)
Destination: /home/kk/jp_claude/RMSD/data/Sen1Floods11/
Run: torch210 container (with network). Re-runs skip existing files.
"""
import os, json, time
import urllib.request
import urllib.parse

BUCKET = "sen1floods11"
DEST = "/home/kk/jp_claude/RMSD/data/Sen1Floods11"
PREFIXES = [
    "v1.1/data/flood_events/HandLabeled/S1Hand/",
    "v1.1/data/flood_events/HandLabeled/S2Hand/",
    "v1.1/data/flood_events/HandLabeled/LabelHand/",
    "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
    "v1.1/splits/",
    "v1.1/Sen1Floods11_Metadata.geojson",
]

def list_objects(prefix):
    items = []
    token = None
    while True:
        url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
               f"?prefix={urllib.parse.quote(prefix)}&maxResults=1000")
        if token:
            url += f"&pageToken={token}"
        with urllib.request.urlopen(url, timeout=60) as r:
            data = json.load(r)
        items += data.get("items", [])
        token = data.get("nextPageToken")
        if not token:
            break
    return items

def download(obj):
    name = obj["name"]
    out = os.path.join(DEST, name)
    size = int(obj["size"])
    if os.path.exists(out) and os.path.getsize(out) == size:
        return False
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = (f"https://storage.googleapis.com/download/storage/v1/b/{BUCKET}/o/"
           f"{urllib.parse.quote(name, safe='')}?alt=media")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(out + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(out + ".part") != size:
                raise IOError(f"size mismatch {name}")
            os.replace(out + ".part", out)
            return True
        except Exception as e:
            if attempt == 3:
                print(f"FAILED {name}: {e}", flush=True)
                return False
            time.sleep(3 * (attempt + 1))

total_files = 0
total_bytes = 0
for prefix in PREFIXES:
    objs = list_objects(prefix)
    sz = sum(int(o["size"]) for o in objs)
    print(f"{prefix}: {len(objs)} files, {sz/1e9:.2f} GB", flush=True)
    done = 0
    for k, o in enumerate(objs):
        if download(o):
            done += 1
        if (k + 1) % 100 == 0:
            print(f"  {prefix}: {k+1}/{len(objs)}", flush=True)
    total_files += len(objs)
    total_bytes += sz
    print(f"  -> complete ({done} new)", flush=True)
print(f"ALL DONE: {total_files} files, {total_bytes/1e9:.2f} GB at {DEST}", flush=True)
