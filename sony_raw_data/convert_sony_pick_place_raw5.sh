#!/usr/bin/env bash
set -euo pipefail

# Sony raw_5: SP1 high/low + SP2 high/low, one four-channel sample per frame.
PYTHON_BIN="${PYTHON_BIN:-/share/project/dy/dy1/envs/act_jack_20260824/bin/python}"
RAW_ROOT="${RAW_ROOT:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_raw_5}"
ROBOT_ROOT="${ROBOT_ROOT:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_robot_raw_5}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_5}"
CONVERTER="/share/project/dy/dy1/code/act_raw/sony_raw_data/convert_sony_pick_place_raw5.py"
START="${START:-1}"
END="${END:-150}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$RAW_ROOT" ]] || { echo "RAW 目录不存在: $RAW_ROOT" >&2; exit 1; }
[[ -d "$ROBOT_ROOT" ]] || { echo "机械臂目录不存在: $ROBOT_ROOT" >&2; exit 1; }
[[ -f "$CONVERTER" ]] || { echo "转换脚本不存在: $CONVERTER" >&2; exit 1; }

args=(
  --raw-root "$RAW_ROOT"
  --robot-root "$ROBOT_ROOT"
  --output-dir "$OUTPUT_DIR"
  --start "$START"
  --end "$END"
  --output-width "${OUTPUT_WIDTH:-224}"
  --output-height "${OUTPUT_HEIGHT:-224}"
  --count-policy "trim"
  --action-alignment "next"
)
if [[ "${OVERWRITE:-0}" == "1" ]]; then args+=(--overwrite); fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then args+=(--dry-run); fi

echo "RAW:      $RAW_ROOT"
echo "ROBOT:    $ROBOT_ROOT"
echo "OUTPUT:   $OUTPUT_DIR"
echo "EPISODES: $START-$END"
exec "$PYTHON_BIN" "$CONVERTER" "${args[@]}"
