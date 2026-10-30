#!/bin/bash
# Stream the remaining batches: DL -> gzip check -> isolated extract -> harvest -> delete.
# Keep only one batch on disk at a time to stay within 1TB.
# Completion marker: $DEST/.harvested_NN
set -u
DEST=/home/kk/jp_claude/RMSD/data/KuroSiwo
STG=/home/kk/jp_claude/RMSD/data/KuroSiwo_staging
SUBSET=/home/kk/jp_claude/RMSD/data/KuroSiwo_subset
HARVEST=/home/kk/jp_claude/RMSD/scripts/harvest_kurosiwo.py
mkdir -p "$STG"

declare -A urls=(
  ["04.tar.gz"]="https://www.dropbox.com/scl/fi/2o3cxwxu79phijxylm99h/04.tar.gz?rlkey=3mcmjrzbf8vmxwl8aacou7vvr&st=yh0e4p72&dl=1"
  ["06.tar.gz"]="https://www.dropbox.com/scl/fi/843fba3poe6nu67og5n6x/06.tar.gz?rlkey=bnxhf9zj0y8mtvjv7k683ywaa&st=p4pg93ko&dl=1"
  ["07.tar.gz"]="https://www.dropbox.com/scl/fi/89kogi7nisinbfa3z2wze/07.tar.gz?rlkey=byduc9q5cndwjphq4r1727xrl&st=a2sgt9dp&dl=1"
  ["08.tar.gz"]="https://www.dropbox.com/scl/fi/sctg1ybqxtzz23018htwx/08.tar.gz?rlkey=6hj887jxezjq5gibqz4zhh7gd&st=9gcgiwdu&dl=1"
  ["09.tar.gz"]="https://www.dropbox.com/scl/fi/425wrfqqc7sy8pbdnkgzw/09.tar.gz?rlkey=ziejzpo2ddgnygpbd1emn1nir&st=irjm2j98&dl=1"
  ["10.tar.gz"]="https://www.dropbox.com/scl/fi/rwc2yvd0g070qicg300de/10.tar.gz?rlkey=12i17si2kzjpey4vgig9pjnv7&st=p2v45gue&dl=1"
)

failed=()
for name in 04.tar.gz 06.tar.gz 07.tar.gz 08.tar.gz 09.tar.gz 10.tar.gz; do
  nn="${name%%.tar.gz}"
  if [ -f "$DEST/.harvested_$nn" ]; then
    echo "SKIP $name (harvested)"
    continue
  fi

  echo "=== free disk before $name:"; df -h / | tail -1

  # With a single process, -c (resume) is safe. Resume after every disconnect,
  # verify gzip once complete; only on verification failure delete and restart.
  ok=0
  for attempt in 1 2 3; do
    echo "=== downloading $name (attempt $attempt, resume)"
    for try in $(seq 1 40); do
      wget -c -q --progress=dot:giga \
           --timeout=60 --read-timeout=120 --waitretry=15 --retry-connrefused \
           "${urls[$name]}" -O "$STG/$name" && break
      echo "  transfer dropped (try $try), resuming in 10s"
      sleep 10
    done
    echo "=== verifying gzip $name"
    if gzip -t "$STG/$name" 2>/dev/null; then
      ok=1; break
    fi
    echo "CORRUPT $name (attempt $attempt) - full restart"
    rm -f "$STG/$name"
  done
  if [ $ok -ne 1 ]; then
    echo "FAILED: $name"; failed+=("$name"); continue
  fi

  echo "=== extracting $name to staging"
  rm -rf "$STG/extract"; mkdir -p "$STG/extract"
  if ! tar -xzf "$STG/$name" -C "$STG/extract"; then
    echo "FAILED extraction: $name"; failed+=("extract:$name")
    rm -rf "$STG/extract" "$STG/$name"; continue
  fi
  rm -f "$STG/$name"          # delete the tar right after extraction

  echo "=== harvesting $name"
  if python3 -u "$HARVEST" --src "$STG/extract" --out "$SUBSET" --min_pflood 1.0 --neg_ratio 1.0; then
    touch "$DEST/.harvested_$nn"
    echo "OK $name"
  else
    echo "FAILED harvest: $name"; failed+=("harvest:$name")
  fi
  rm -rf "$STG/extract"       # delete the extracted files too, then move on
done

rm -rf "$STG"
echo "STREAM DONE. failed: ${failed[*]:-none}"
du -sh "$SUBSET"; df -h / | tail -1
