#!/usr/bin/env bash
# Download the data of the paper into DATA_ROOT (default ./data/raw).
#
#   scripts/download.sh uavsar [s1|s2|ann|all]   # UAVSAR SDelta_23518_02, public, no login
#   scripts/download.sh opera                    # OPERA CSLC-S1, needs an Earthdata token
#   scripts/download.sh stores [s1|s2|all]       # the paper's UAVSAR stores (Zenodo), see below
#
# Resumable: complete files are skipped, partial ones resumed. Sizes: one UAVSAR 1x1 SLC is
# 5.3 GB; segment s1 (204 SLCs, 3 pols x 68 acquisitions) is about 1.1 TB, s2 the same.
# `stores` skips stage 1a: it fetches the VV scan stores of both segments (18 + 19 GB tars,
# doi:10.5281/zenodo.24414662) into $RUNS_ROOT/large/runs (manifests in $RUNS_ROOT/runs), checks
# the tar SHA-256, then the content hashes of data/stores.json. Extracting needs the tar size
# again in free space.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
root="${DATA_ROOT:-$here/data/raw}"

case "${1:-}" in
  uavsar)
    dest="$root/uavsar/SDelta_23518"; mkdir -p "$dest"
    case "${2:-all}" in
      s1)  lists=(uavsar_ann.txt uavsar_s1.txt) ;;
      s2)  lists=(uavsar_ann.txt uavsar_s2.txt) ;;
      ann) lists=(uavsar_ann.txt) ;;
      all) lists=(uavsar_ann.txt uavsar_s1.txt uavsar_s2.txt) ;;
      *) echo "unknown segment ${2}" >&2; exit 1 ;;
    esac
    for l in "${lists[@]}"; do
      while read -r url; do
        [ -z "$url" ] && continue
        (cd "$dest" && curl -sSfL --retry 5 -C - -O "$url") || echo "FAIL $url" >&2
      done < "$here/data/urls/$l"
    done ;;
  opera)
    token_file="${EARTHDATA_TOKEN_FILE:-$HOME/.config/earthdata/token}"
    [ -f "$token_file" ] || { echo "Earthdata token not found in $token_file" >&2; exit 1; }
    dest="$root/sentinel1/OPERA_CSLC"; mkdir -p "$dest"
    cat "$here/data/urls/opera_cslc.txt" "$here/data/urls/opera_cslc_seg2.txt" |
    while read -r sub url; do
      [ -z "$url" ] && continue
      f="$dest/$sub/$(basename "$url")"; mkdir -p "$dest/$sub"
      [ -f "$f" ] && continue
      curl -sSfL --retry 5 -C - -o "$f.part" \
        -H @<(printf 'Authorization: Bearer %s' "$(cat "$token_file")") "$url" \
        && mv "$f.part" "$f" || echo "FAIL $url" >&2
    done ;;
  stores)
    rec="https://zenodo.org/records/24414662/files"
    dest="${RUNS_ROOT:-$here/runs}/large/runs"; small="${RUNS_ROOT:-$here/runs}/runs"
    mkdir -p "$dest"
    case "${2:-all}" in
      s1) segs=(1) ;; s2) segs=(2) ;; all) segs=(1 2) ;;
      *) echo "unknown segment ${2}" >&2; exit 1 ;;
    esac
    if command -v sha256sum > /dev/null; then sha=(sha256sum); else sha=(shasum -a 256); fi
    curl -sSfL --retry 5 -o "$dest/SHA256SUMS.zenodo" "$rec/SHA256SUMS?download=1"
    for s in "${segs[@]}"; do
      run=$(cd "$here" && uv run --no-sync python -c "import json; print(json.load(open(
        'data/stores.json'))['uavsar-scan-s$s']['vv_rebuild_run'])")
      if [ ! -d "$dest/$run/scan.zarr" ]; then
        echo "downloading $run.tar"
        (cd "$dest" && curl -fL --retry 5 -C - -o "$run.tar" "$rec/$run.tar?download=1")
        (cd "$dest" && grep " $run.tar\$" SHA256SUMS.zenodo | "${sha[@]}" -c -)
        # the tar holds the run directory twice: the store (large/runs) and, read-only, the
        # manifest and log (small outputs, runs/)
        tar -xf "$dest/$run.tar" -C "$dest" --exclude "$run/manifest.json" --exclude "$run/log.txt"
        mkdir -p "$small"
        tar -xf "$dest/$run.tar" -C "$small" "$run/manifest.json" "$run/log.txt"
        rm "$dest/$run.tar"
      fi
      (cd "$here" && uv run python experiments/stores/verify_store.py "$dest/$run/scan.zarr" \
        --check "uavsar-scan-s$s")
    done ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
