#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GEE authentication connectivity check (prints no secrets)."""
import os, json, sys
p = os.path.expanduser("~/.config/earthengine/credentials")
print("credentials file:", "OK" if os.path.exists(p) else "MISSING")
if os.path.exists(p):
    d = json.load(open(p))
    print("  keys present:", sorted(d.keys()))
    print("  project in file:", d.get("project") or d.get("quota_project_id") or "(none)")

import ee
proj = sys.argv[1] if len(sys.argv) > 1 else None
try:
    ee.Initialize(project=proj) if proj else ee.Initialize()
    n = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate("2020-10-13", "2020-10-15").size()
    print("ee.Initialize OK  (project=%s)" % (proj or "default"))
    print("S1_GRD scenes on 2020-10-13/14 worldwide:", n.getInfo())
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED").filterDate("2020-10-13", "2020-10-15").size()
    print("S2_HARMONIZED scenes same window:", s2.getInfo())
except Exception as e:
    print("ee.Initialize FAILED:", str(e)[:500])
