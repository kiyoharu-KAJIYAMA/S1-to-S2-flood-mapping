#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch C2S-MS Floods from AWS S3 (radiant-mlhub/c2smsfloods, anonymous).
61,203 files ~10.7GB. 8 parallel workers, skip existing files, atomic .part rename.
Destination: /home/kk/jp_claude/RMSD/data/C2SMSFloods/
Run: WSL native python.
"""
import os, time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

S3 = "https://radiant-mlhub.s3.us-west-2.amazonaws.com"
PREFIX = "c2smsfloods/"
DEST = "/home/kk/jp_claude/RMSD/data/C2SMSFloods"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

def list_all():
    items, token = [], None
    while True:
        url = f"{S3}/?list-type=2&prefix={PREFIX}&max-keys=1000"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(f"{NS}Contents"):
            items.append((c.find(f"{NS}Key").text, int(c.find(f"{NS}Size").text)))
        if root.find(f"{NS}IsTruncated").text != "true":
            break
        token = root.find(f"{NS}NextContinuationToken").text
    return items

def download(args):
    key, size = args
    rel = key[len(PREFIX):]
    if not rel:
        return 0
    out = os.path.join(DEST, rel)
    if os.path.exists(out) and os.path.getsize(out) == size:
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f"{S3}/{urllib.parse.quote(key)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(out + ".part", "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(out + ".part") != size:
                raise IOError("size mismatch")
            os.replace(out + ".part", out)
            return 1
        except Exception as e:
            if attempt == 3:
                print(f"FAILED {key}: {e}", flush=True)
                return -1
            time.sleep(2 * (attempt + 1))

items = list_all()
print(f"listing: {len(items)} files, {sum(s for _, s in items)/1e9:.2f} GB", flush=True)
done = fail = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    for k, r in enumerate(ex.map(download, items)):
        if r == -1:
            fail += 1
        else:
            done += 1
        if (k + 1) % 2000 == 0:
            print(f"  {k+1}/{len(items)} ({(k+1)/(time.time()-t0):.0f} files/s)", flush=True)
print(f"ALL DONE: {done} ok, {fail} failed -> {DEST}", flush=True)
