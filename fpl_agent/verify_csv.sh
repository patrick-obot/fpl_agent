#!/usr/bin/env bash
set -euo pipefail

INCOMING="${1:-data/projected_points.csv.tmp}"
TARGET="${2:-data/projected_points.csv}"
MIN_BYTES="${MIN_BYTES:-20000}"
MIN_LINES="${MIN_LINES:-100}"
MIN_PLAYER_ROWS="${MIN_PLAYER_ROWS:-20}"

fail() {
  echo "[verify_csv] ERROR: $1" >&2
  exit 1
}

[ -f "$INCOMING" ] || fail "Incoming file not found: $INCOMING"

BYTES=$(wc -c < "$INCOMING" | tr -d ' ')
LINES=$(wc -l < "$INCOMING" | tr -d ' ')

[ "$BYTES" -ge "$MIN_BYTES" ] || fail "File too small: ${BYTES} bytes (min ${MIN_BYTES})"
[ "$LINES" -ge "$MIN_LINES" ] || fail "Too few lines: ${LINES} (min ${MIN_LINES})"

HEADER=$(head -n 1 "$INCOMING" | tr -d '\r')
for COL in Pos ID Name Team; do
  echo "$HEADER" | grep -Eq "(^|,)${COL}(,|$)" || fail "Missing required header column: ${COL}"
done

PLAYER_ROWS=$(awk -F',' 'NR > 1 && $1 ~ /^[GDMF]$/ && $2 ~ /^[0-9]+$/ && length($3) > 0 { c++ } END { print c+0 }' "$INCOMING")
[ "$PLAYER_ROWS" -ge "$MIN_PLAYER_ROWS" ] || fail "Too few valid player rows: ${PLAYER_ROWS} (min ${MIN_PLAYER_ROWS})"

mkdir -p "$(dirname "$TARGET")"
if [ -f "$TARGET" ]; then
  cp -f "$TARGET" "${TARGET}.bak"
fi

mv -f "$INCOMING" "$TARGET"
echo "[verify_csv] OK: promoted ${TARGET} (${BYTES} bytes, ${LINES} lines, ${PLAYER_ROWS} player rows)"
