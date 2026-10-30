#!/bin/bash
# Fetch the Kuro Siwo GRD dataset (Dropbox, dl=1).
# Dropbox ignores range requests (resume) and corrupts the file by appending, so
#   do not use -c: fetch from scratch every time -> verify gzip integrity -> extract.
# 00-03 are already extracted (.done marker) and are skipped.
set -u
DEST=/home/kk/jp_claude/RMSD/data/KuroSiwo
mkdir -p "$DEST"

declare -A urls=(
  ["catalogue.gpkg"]="https://www.dropbox.com/scl/fi/wu6nvj73cz4h7k3gxpzx6/catalogue.gpkg?rlkey=hsij2o0k60r2n0z6z4d2ngww9&st=0zjqhzgx&dl=1"
  ["00.tar.gz"]="https://www.dropbox.com/scl/fi/19mm9v5pnd5yor8b15alj/00.tar.gz?rlkey=f7qrqgv7h7z9j6r595xz1720e&st=k4ptwanm&dl=1"
  ["01.tar.gz"]="https://www.dropbox.com/scl/fi/pjlgcqc3fm8lx97vdfjyn/01.tar.gz?rlkey=bvue0u3jgovc3qewkh464uyog&st=qfsa601d&dl=1"
  ["02.tar.gz"]="https://www.dropbox.com/scl/fi/7tt843025s8hqi00xpofx/02.tar.gz?rlkey=8vbfh3qc2h7zmzjem0c14lljh&st=xda2zf5j&dl=1"
  ["03.tar.gz"]="https://www.dropbox.com/scl/fi/v7997kv11cc8ptj1vstim/03.tar.gz?rlkey=yj0qspgmlmohaf728a94curqf&st=6qmn63y7&dl=1"
  ["04.tar.gz"]="https://www.dropbox.com/scl/fi/2o3cxwxu79phijxylm99h/04.tar.gz?rlkey=3mcmjrzbf8vmxwl8aacou7vvr&st=yh0e4p72&dl=1"
  ["05.tar.gz"]="https://www.dropbox.com/scl/fi/q1jy4ep4j6bj38dva2es3/05.tar.gz?rlkey=g46498pgox03it080p83xhx8g&st=haxheepl&dl=1"
  ["06.tar.gz"]="https://www.dropbox.com/scl/fi/843fba3poe6nu67og5n6x/06.tar.gz?rlkey=bnxhf9zj0y8mtvjv7k683ywaa&st=p4pg93ko&dl=1"
  ["07.tar.gz"]="https://www.dropbox.com/scl/fi/89kogi7nisinbfa3z2wze/07.tar.gz?rlkey=byduc9q5cndwjphq4r1727xrl&st=a2sgt9dp&dl=1"
  ["08.tar.gz"]="https://www.dropbox.com/scl/fi/sctg1ybqxtzz23018htwx/08.tar.gz?rlkey=6hj887jxezjq5gibqz4zhh7gd&st=9gcgiwdu&dl=1"
  ["09.tar.gz"]="https://www.dropbox.com/scl/fi/425wrfqqc7sy8pbdnkgzw/09.tar.gz?rlkey=ziejzpo2ddgnygpbd1emn1nir&st=irjm2j98&dl=1"
  ["10.tar.gz"]="https://www.dropbox.com/scl/fi/rwc2yvd0g070qicg300de/10.tar.gz?rlkey=12i17si2kzjpey4vgig9pjnv7&st=p2v45gue&dl=1"
)

failed=()
for name in catalogue.gpkg 00.tar.gz 01.tar.gz 02.tar.gz 03.tar.gz 04.tar.gz 05.tar.gz 06.tar.gz 07.tar.gz 08.tar.gz 09.tar.gz 10.tar.gz; do
  if [ -f "$DEST/.done_$name" ]; then
    echo "SKIP $name (done)"
    continue
  fi

  ok=0
  for attempt in 1 2 3; do
    echo "=== downloading $name (attempt $attempt)"
    rm -f "$DEST/$name"                       # start from scratch every time to avoid append corruption
    wget -q --show-progress --progress=dot:giga \
         --timeout=60 --read-timeout=60 --tries=5 --waitretry=15 --retry-connrefused \
         "${urls[$name]}" -O "$DEST/$name"
    if [ $? -ne 0 ]; then
      echo "download error $name (attempt $attempt), retrying"
      continue
    fi
    if [[ "$name" == *.tar.gz ]]; then
      echo "=== verifying gzip integrity $name"
      if ! gzip -t "$DEST/$name" 2>/dev/null; then
        echo "CORRUPT $name (attempt $attempt), redownloading"
        rm -f "$DEST/$name"
        continue
      fi
    fi
    ok=1
    break
  done

  if [ $ok -ne 1 ]; then
    echo "FAILED after retries: $name"
    failed+=("$name")
    rm -f "$DEST/$name"
    continue
  fi

  if [[ "$name" == *.tar.gz ]]; then
    echo "=== extracting $name"
    if tar -xzf "$DEST/$name" -C "$DEST"; then
      rm "$DEST/$name"
      touch "$DEST/.done_$name"
      echo "OK $name"
    else
      echo "FAILED extraction: $name"
      failed+=("extract:$name")
    fi
  else
    touch "$DEST/.done_$name"
    echo "OK $name"
  fi
done

echo "ALL DONE. failed: ${failed[*]:-none}"
du -sh "$DEST"
