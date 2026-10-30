#!/bin/bash
# Fetch and extract the FloodPlanet Inundation Dataset (Zenodo 15238572, CC-BY-4.0, 1.88GB).
set -u
DEST=/home/kk/jp_claude/RMSD/data/FloodPlanet
mkdir -p "$DEST"
URL="https://zenodo.org/api/records/15238572/files/FloodPlanet.zip/content"

if [ ! -f "$DEST/FloodPlanet.zip" ]; then
  wget -c -q --show-progress --progress=dot:giga "$URL" -O "$DEST/FloodPlanet.zip"
  if [ $? -ne 0 ]; then echo "FAILED download"; exit 1; fi
fi
echo "=== extracting"
python3 -c "
import zipfile
with zipfile.ZipFile('$DEST/FloodPlanet.zip') as z:
    z.extractall('$DEST')
print('extracted', len(z.namelist()), 'entries')
"
echo "ALL DONE"
du -sh "$DEST"
