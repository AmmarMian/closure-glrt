#!/usr/bin/env bash
# Download the data of the paper into DATA_ROOT (default ./data/raw).
#
#   scripts/download.sh uavsar [s1|s2|ann|all]   # UAVSAR SDelta_23518_02, public, no login
#   scripts/download.sh opera                    # OPERA CSLC-S1, needs an Earthdata token
#
# Resumable: complete files are skipped, partial ones resumed. Sizes: one UAVSAR 1x1 SLC is
# 5.3 GB; segment s1 (204 SLCs, 3 pols x 68 acquisitions) is about 1.1 TB, s2 the same.
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
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
