#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/share/project/dy/dy1/code/act_raw_5"
PYTHON_BIN="/share/project/dy/dy1/envs/act_jack_20260824/bin/python"
DATA_DIR="${DATA_DIR:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}"
TASK_NAME="${TASK_NAME:-pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_050_step30000}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-policy_step_30000_seed_42.ckpt}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/run/$TASK_NAME/${CHECKPOINT_NAME%.ckpt}}"
EVALUATOR="$PROJECT_ROOT/scripts/eval_open_loop_raw10_quat.py"
START="${EPISODE_START:-51}"
END="${EPISODE_END:-60}"

CHECKPOINT="$CHECKPOINT_ROOT/$TASK_NAME/$CHECKPOINT_NAME"
STATS="$CHECKPOINT_ROOT/$TASK_NAME/dataset_stats.pkl"
[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用：$PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "数据目录不存在：$DATA_DIR" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "评估程序不存在：$EVALUATOR" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "权重不存在：$CHECKPOINT" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "统计文件不存在：$STATS" >&2; exit 1; }

mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_VISIBLE_DEVICES=all CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export ACT_STARTOUCH_SINGLE_STATE_DIM=8 ACT_STARTOUCH_SINGLE_ACTION_DIM=8
export DEVICE=cuda PYTHONUNBUFFERED=1

CSV="$RUN_DIR/open_loop_eval_${START}_${END}_sony_raw5_linear_quat.csv"
JSON="$RUN_DIR/open_loop_eval_${START}_${END}_sony_raw5_linear_quat.summary.json"
MD="$RUN_DIR/open_loop_eval_${START}_${END}_sony_raw5_linear_quat.md"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$EVALUATOR" \
  --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_ROOT" \
  --task "$TASK_NAME" --checkpoint-name "$CHECKPOINT_NAME" \
  --episode-start "$START" --episode-end "$END" \
  --num-queries "${NUM_QUERIES:-80}" --batch-size "${BATCH_SIZE:-64}" \
  --raw-black-level 0 --raw-white-level 4095 --raw-gamma 1.0 \
  --device cuda --output-csv "$CSV" --output-json "$JSON" --output-markdown "$MD"
