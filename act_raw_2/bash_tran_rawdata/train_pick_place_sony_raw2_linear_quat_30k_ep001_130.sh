#!/usr/bin/env bash
set -euo pipefail

# Sony raw_2 four-channel linear RAW ACT training.
# Episodes 001-130 are used for training/internal validation; 131-150 are
# kept untouched for the automatic open-loop evaluation after training.

PROJECT_ROOT="/share/project/dy/dy1/code/act_raw/act_raw_2"
PYTHON_BIN="${PYTHON_BIN:-/share/project/dy/dy1/envs/act_jack_20260824/bin/python}"
DATA_DIR="${DATA_DIR:-/share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
TASK_NAME="${TASK_NAME:-pick_place_sony_raw2_sp1_sp2_linear_quat_ep001_130_step40000}"
EVALUATOR="${PROJECT_ROOT}/scripts/eval_open_loop_raw10_quat.py"
TORCH_HOME="${TORCH_HOME:-${PROJECT_ROOT}/.torch}"

GPU_ID="${GPU_ID:-0}"
TARGET_STEPS="${TARGET_STEPS:-40000}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-16}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-16}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-5000}"
VALIDATION_INTERVAL_EPOCHS="${VALIDATION_INTERVAL_EPOCHS:-50}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
VAL_WORKERS="${VAL_WORKERS:-2}"
TRAIN_PREFETCH="${TRAIN_PREFETCH:-4}"
VAL_PREFETCH="${VAL_PREFETCH:-1}"
RAW_BLACK_LEVEL="${RAW_BLACK_LEVEL:-0}"
RAW_WHITE_LEVEL="${RAW_WHITE_LEVEL:-4095}"
RAW_GAMMA="${RAW_GAMMA:-1.0}"
TRAIN_START=1
TRAIN_END=130
EVAL_START=131
EVAL_END=150

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用: $PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "数据目录不存在: $DATA_DIR" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "评估程序不存在: $EVALUATOR" >&2; exit 1; }

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_VISIBLE_DEVICES=all CUDA_VISIBLE_DEVICES="$GPU_ID"

PREFLIGHT="$($PYTHON_BIN - "$DATA_DIR" "$BATCH_SIZE_TRAIN" "$TARGET_STEPS" <<'PY'
import glob, h5py, math, os, re, sys

data_dir, batch_size, target_steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
episodes = {}
for path in glob.glob(os.path.join(data_dir, "episode_*.hdf5")):
    m = re.fullmatch(r"episode_(\d+)\.hdf5", os.path.basename(path))
    if m: episodes[int(m.group(1))] = path
expected = set(range(1, 151))
missing = sorted(expected - set(episodes))
if missing: raise SystemExit(f"缺少 episode: {missing}")
bad = []
lengths = []
for eid in range(1, 151):
    path = episodes[eid]
    with h5py.File(path, "r") as root:
        required = ("action", "state", "observations/qpos", "observations/images/front")
        missing_keys = [k for k in required if k not in root]
        if missing_keys: raise SystemExit(f"{path} 缺少: {missing_keys}")
        action = root["action"].shape
        qpos = root["observations/qpos"].shape
        state = root["state"].shape
        images = root["observations/images/front"].shape
        if action != qpos or action != state or len(action) != 2 or action[1] != 8:
            raise SystemExit(f"{path}: action={action}, qpos={qpos}, state={state}")
        if images != (action[0], 4, 224, 224):
            raise SystemExit(f"{path}: images={images}, 期待 (T,4,224,224)")
        encoding = root.attrs.get("image_encoding", "")
        channels = root.attrs.get("channel_names", "")
        if isinstance(encoding, bytes): encoding = encoding.decode()
        if isinstance(channels, bytes): channels = channels.decode()
        if encoding != "sony_bin_uint16": raise SystemExit(f"{path}: image_encoding={encoding!r}")
        if channels != "sp1_high,sp1_low,sp2_high,sp2_low":
            raise SystemExit(f"{path}: channel_names={channels!r}")
        lengths.append(action[0])
train_count = 130
inner_train = min(max(1, int(0.95 * train_count)), train_count - 1)
inner_val = train_count - inner_train
batches = math.ceil(inner_train / batch_size)
if target_steps % batches:
    raise SystemExit(f"TARGET_STEPS={target_steps} 不能被每 epoch {batches} 个 batch 整除")
print(max(lengths), train_count, 20, inner_train, inner_val, batches, target_steps // batches)
PY
)"
read -r EPISODE_LEN TRAIN_COUNT EVAL_COUNT INNER_TRAIN INNER_VAL STEPS_PER_EPOCH NUM_EPOCHS <<< "$PREFLIGHT"

echo "Project:             $PROJECT_ROOT"
echo "Dataset:             $DATA_DIR"
echo "Train episodes:      $TRAIN_START-$TRAIN_END ($TRAIN_COUNT)"
echo "Open-loop episodes:  $EVAL_START-$EVAL_END ($EVAL_COUNT)"
echo "Episode length max:  $EPISODE_LEN"
echo "State/action:        8D [x,y,z,qx,qy,qz,qw,gripper]"
echo "Image:               Sony uint16 CHW, 4 channels [sp1_high,sp1_low,sp2_high,sp2_low]"
echo "RAW scaling:         black=$RAW_BLACK_LEVEL white=$RAW_WHITE_LEVEL gamma=$RAW_GAMMA"
echo "Steps:               $TARGET_STEPS ($STEPS_PER_EPOCH/epoch, $NUM_EPOCHS epochs)"
echo "Checkpoint:          $CHECKPOINT_ROOT/$TASK_NAME"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1，只完成数据检查，不启动训练和评估。"
    exit 0
fi

TASK_CHECKPOINT_DIR="$CHECKPOINT_ROOT/$TASK_NAME"
if [[ -d "$TASK_CHECKPOINT_DIR" ]] && find "$TASK_CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q . && [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
    echo "Checkpoint 目录非空：$TASK_CHECKPOINT_DIR；如需继续请设置 ALLOW_EXISTING=1" >&2
    exit 1
fi
mkdir -p "$TORCH_HOME/hub/checkpoints" "$CHECKPOINT_ROOT"
if [[ -f "$TORCH_HOME/resnet18-f37072fd.pth" && ! -e "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" ]]; then
    ln -s "$TORCH_HOME/resnet18-f37072fd.pth" "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
fi

export TORCH_HOME DEVICE=cuda PYTHONUNBUFFERED=1
export ACT_STARTOUCH_SINGLE_DATA_DIR="$DATA_DIR"
export ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR="$CHECKPOINT_ROOT"
export ACT_STARTOUCH_SINGLE_EPISODE_RANGES="${TRAIN_START}-${TRAIN_END}"
export ACT_STARTOUCH_SINGLE_EPISODE_LEN="$EPISODE_LEN"
export ACT_STARTOUCH_SINGLE_NUM_EPOCHS="$NUM_EPOCHS"
export ACT_STARTOUCH_SINGLE_TARGET_STEPS="$TARGET_STEPS"
export ACT_STARTOUCH_SINGLE_STATE_DIM=8 ACT_STARTOUCH_SINGLE_ACTION_DIM=8
export ACT_STARTOUCH_SINGLE_BATCH_SIZE_TRAIN="$BATCH_SIZE_TRAIN" ACT_STARTOUCH_SINGLE_BATCH_SIZE_VAL="$BATCH_SIZE_VAL"
export ACT_STARTOUCH_SINGLE_CHECKPOINT_INTERVAL_STEPS="$CHECKPOINT_INTERVAL_STEPS"
export ACT_STARTOUCH_SINGLE_VALIDATION_INTERVAL_EPOCHS="$VALIDATION_INTERVAL_EPOCHS"
export ACT_STARTOUCH_SINGLE_TRAIN_WORKERS="$TRAIN_WORKERS" ACT_STARTOUCH_SINGLE_VAL_WORKERS="$VAL_WORKERS"
export ACT_STARTOUCH_SINGLE_TRAIN_PREFETCH="$TRAIN_PREFETCH" ACT_STARTOUCH_SINGLE_VAL_PREFETCH="$VAL_PREFETCH"
export ACT_STARTOUCH_SINGLE_WANDB=0

cd "$PROJECT_ROOT"
"$PYTHON_BIN" train.py --task "$TASK_NAME" --data-format raw10_hdf5 \
    --raw-black-level "$RAW_BLACK_LEVEL" --raw-white-level "$RAW_WHITE_LEVEL" --raw-gamma "$RAW_GAMMA"

CHECKPOINT="$TASK_CHECKPOINT_DIR/policy_last.ckpt"
STATS="$TASK_CHECKPOINT_DIR/dataset_stats.pkl"
[[ -f "$CHECKPOINT" ]] || { echo "训练结束但找不到 checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -f "$STATS" ]] || { echo "训练结束但找不到 dataset_stats.pkl: $STATS" >&2; exit 1; }

RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/run/$TASK_NAME/policy_last/open_loop_eval_${EVAL_START}_${EVAL_END}}"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" "$EVALUATOR" \
    --data-dir "$DATA_DIR" --checkpoint-dir "$CHECKPOINT_ROOT" --task "$TASK_NAME" \
    --checkpoint-name policy_last.ckpt --episode-start "$EVAL_START" --episode-end "$EVAL_END" \
    --num-queries 80 --batch-size "${EVAL_BATCH_SIZE:-64}" \
    --raw-black-level "$RAW_BLACK_LEVEL" --raw-white-level "$RAW_WHITE_LEVEL" --raw-gamma "$RAW_GAMMA" \
    --device cuda --output-csv "$RUN_DIR/open_loop_eval.csv" \
    --output-json "$RUN_DIR/open_loop_eval.summary.json" --output-markdown "$RUN_DIR/open_loop_eval.md"
