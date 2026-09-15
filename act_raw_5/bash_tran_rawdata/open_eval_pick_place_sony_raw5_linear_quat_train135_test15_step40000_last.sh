#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/share/project/dy/dy1/code/act_raw/act_raw_5"
PYTHON_BIN="${PYTHON_BIN:-/share/project/dy/dy1/envs/act_jack_20260824/bin/python}"
DATA_DIR="${DATA_DIR:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_5}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
TASK_NAME="${TASK_NAME:-pick_place_sony_raw5_sp1_sp2_linear_quat_train135_test15_every10_step40000}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-policy_last.ckpt}"
EVALUATOR="${PROJECT_ROOT}/scripts/eval_open_loop_raw_quat.py"
TEST_IDS=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150)
RAW_BLACK_LEVEL="${RAW_BLACK_LEVEL:-0}"
RAW_WHITE_LEVEL="${RAW_WHITE_LEVEL:-4095}"
RAW_GAMMA="${RAW_GAMMA:-1.0}"
GPU_ID="${GPU_ID:-1}"

CHECKPOINT="$CHECKPOINT_ROOT/$TASK_NAME/$CHECKPOINT_NAME"
STATS="$CHECKPOINT_ROOT/$TASK_NAME/dataset_stats.pkl"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/run/$TASK_NAME/policy_last/open_loop_eval_test_every10_15eps}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "数据目录不存在: $DATA_DIR" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "评估程序不存在: $EVALUATOR" >&2; exit 1; }
[[ -f "$CHECKPOINT" ]] || { echo "权重不存在: $CHECKPOINT" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "统计文件不存在: $STATS" >&2; exit 1; }

mkdir -p "$RUN_DIR"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_VISIBLE_DEVICES=all CUDA_VISIBLE_DEVICES="$GPU_ID"
export ACT_STARTOUCH_SINGLE_STATE_DIM=8 ACT_STARTOUCH_SINGLE_ACTION_DIM=8
export DEVICE=cuda PYTHONUNBUFFERED=1 TORCH_HOME="${TORCH_HOME:-$PROJECT_ROOT/.torch}"

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" "$EVALUATOR" \
    --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_ROOT" --task "$TASK_NAME" \
    --checkpoint-name "$CHECKPOINT_NAME" --episode-ids "${TEST_IDS[@]}" \
    --num-queries 80 --batch-size "${BATCH_SIZE:-64}" \
    --raw-black-level "$RAW_BLACK_LEVEL" --raw-white-level "$RAW_WHITE_LEVEL" --raw-gamma "$RAW_GAMMA" \
    --device cuda --output-csv "$RUN_DIR/open_loop_eval.csv" \
    --output-json "$RUN_DIR/open_loop_eval.summary.json" \
    --output-markdown "$RUN_DIR/open_loop_eval.md"
