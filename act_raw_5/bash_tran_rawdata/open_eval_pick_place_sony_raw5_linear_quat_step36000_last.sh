#!/usr/bin/env bash
set -euo pipefail

# Open-loop evaluation for the raw5 linear four-channel Sony model.
# Training/evaluation image path:
# uint16 CHW -> clamp(raw / 4095, 0, 1) -> no gamma -> mean=0/std=1.

PROJECT_ROOT="/share/project/dy/dy1/code/act_raw_5"
PYTHON_BIN="/share/project/dy/dy1/envs/act_jack_20260824/bin/python"
DATA_DIR="${DATA_DIR:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
TASK_NAME="pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_050_step36000"
CHECKPOINT_NAME="policy_last.ckpt"
EVALUATOR="$PROJECT_ROOT/scripts/eval_open_loop_raw10_quat.py"
TORCH_HOME="$PROJECT_ROOT/.torch"

START="${EPISODE_START:-51}"
END="${EPISODE_END:-60}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_QUERIES=80
RAW_BLACK_LEVEL=0
RAW_WHITE_LEVEL=4095
RAW_GAMMA=1.0

RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/run/$TASK_NAME/policy_last/open_loop_eval_${START}_${END}}"
CHECKPOINT="$CHECKPOINT_ROOT/$TASK_NAME/$CHECKPOINT_NAME"
STATS="$CHECKPOINT_ROOT/$TASK_NAME/dataset_stats.pkl"

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用：$PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "数据目录不存在：$DATA_DIR" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "评估程序不存在：$EVALUATOR" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "权重不存在：$CHECKPOINT" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "统计文件不存在：$STATS" >&2; exit 1; }

"$PYTHON_BIN" - "$DATA_DIR" "$START" "$END" <<'PY'
import sys
from pathlib import Path

import h5py

data_dir = Path(sys.argv[1])
start, end = map(int, sys.argv[2:4])
if start > end:
    raise SystemExit(f"episode 范围错误：{start}>{end}")

selected = []
for path in data_dir.glob("episode_*.hdf5"):
    try:
        episode = int(path.stem.rsplit("_", 1)[1])
    except ValueError:
        continue
    if start <= episode <= end:
        selected.append((episode, path))
selected.sort()

found = {episode for episode, _ in selected}
missing = sorted(set(range(start, end + 1)) - found)
if missing:
    raise SystemExit(f"缺少测试 episode：{missing}")

for episode, path in selected:
    with h5py.File(path, "r") as root:
        images = root["observations/images/front"]
        qpos = root["observations/qpos"]
        action = root["action"]
        if len(images.shape) != 4 or images.shape[1:] != (4, 224, 224):
            raise SystemExit(f"{path}: images={images.shape}，期待 (T,4,224,224)")
        if qpos.shape != action.shape or action.shape != (images.shape[0], 8):
            raise SystemExit(
                f"{path}: images={images.shape}, qpos={qpos.shape}, action={action.shape}"
            )
        attrs = root.attrs
        encoding = attrs.get("image_encoding", "")
        layout = attrs.get("image_layout", "")
        channels = attrs.get("channel_names", "")
        for name, value in (("encoding", encoding), ("layout", layout), ("channels", channels)):
            if isinstance(value, bytes):
                if name == "encoding": encoding = value.decode()
                elif name == "layout": layout = value.decode()
                else: channels = value.decode()
        if encoding != "sony_bin_uint16" or layout != "CHW":
            raise SystemExit(f"{path}: encoding={encoding!r}, layout={layout!r}")
        expected = "sp1_low,sp1_high,sp2_low,sp2_high"
        if channels != expected:
            raise SystemExit(f"{path}: channel_names={channels!r}，期待 {expected!r}")
        if "raw_black_level" in attrs and int(attrs["raw_black_level"]) != 0:
            raise SystemExit(f"{path}: raw_black_level 不是 0")
        if "raw_white_level" in attrs and int(attrs["raw_white_level"]) != 4095:
            raise SystemExit(f"{path}: raw_white_level 不是 4095")

print(f"数据检查通过：episode {start}-{end}，{len(selected)} 集")
PY

mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_VISIBLE_DEVICES=all
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export ACT_STARTOUCH_SINGLE_STATE_DIM=8
export ACT_STARTOUCH_SINGLE_ACTION_DIM=8
export DEVICE=cuda
export PYTHONUNBUFFERED=1
export TORCH_HOME

CSV="$RUN_DIR/open_loop_eval.csv"
JSON="$RUN_DIR/open_loop_eval.summary.json"
MD="$RUN_DIR/open_loop_eval.md"

echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATA_DIR"
echo "Episodes:   $START-$END"
echo "RAW:        black=$RAW_BLACK_LEVEL white=$RAW_WHITE_LEVEL gamma=$RAW_GAMMA"
echo "Output:     $RUN_DIR"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$EVALUATOR" \
  --data-dir "$DATA_DIR" \
  --checkpoint-dir "$CHECKPOINT_ROOT" \
  --task "$TASK_NAME" \
  --checkpoint-name "$CHECKPOINT_NAME" \
  --episode-start "$START" \
  --episode-end "$END" \
  --num-queries "$NUM_QUERIES" \
  --batch-size "$BATCH_SIZE" \
  --raw-black-level "$RAW_BLACK_LEVEL" \
  --raw-white-level "$RAW_WHITE_LEVEL" \
  --raw-gamma "$RAW_GAMMA" \
  --device cuda \
  --output-csv "$CSV" \
  --output-json "$JSON" \
  --output-markdown "$MD"
