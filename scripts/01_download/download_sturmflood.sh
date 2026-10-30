#!/bin/bash
# Fetch STURM-Flood (Zenodo 12748982, CC-BY-4.0, Dataset.zip 3.99GB) + MD5 check + extract.
# To stop it, always kill by PID (pkill -f matches itself).
set -u
DST=/home/kk/jp_claude/RMSD/data/STURMFlood
EXP=0e4172e74a4cf2e4c608c14d1588d1d9
WANT=3985215183
mkdir -p "$DST"
cd "$DST" || exit 3

echo "=== STURM-Flood download start $(date '+%F %T') ==="
wget -c --tries=10 --timeout=60 --waitretry=30 --retry-connrefused \
     -o "$DST/wget.log" https://zenodo.org/records/12748983/files/Dataset.zip -O Dataset.zip
echo "[dl] exit=$? size=$(stat -c %s Dataset.zip) want=$WANT $(date '+%F %T')"

got=$(md5sum Dataset.zip | cut -d' ' -f1)
if [ "$got" != "$EXP" ]; then
    echo "MD5 FAIL got=$got exp=$EXP"; exit 1
fi
echo "MD5 OK Dataset.zip"

mkdir -p raw
unzip -oq Dataset.zip -d raw/ && echo "extracted" || { echo "EXTRACT FAIL"; exit 1; }
du -sh raw
find raw -maxdepth 3 -type d | head -30
echo "=== STURM-Flood done $(date '+%F %T') ==="
