#!/usr/bin/env bash
set -euo pipefail

# 暗光 RAW10 + 8D quaternion 单卡训练。
# 训练：实际存在的 episode 1-100（包含 67/70）；评估预留 101-125。

PROJECT_ROOT="/share/project/dy/dy1/code/flexiv_act_raw"
DATA_DIR="${DATA_DIR:-/share/project/dy/dy1/data/rawdata/pick_place_dark_robot_raw10_hdf5}"
PYTHON_BIN="/share/project/dy/dy1/envs/act_jack_20260824/bin/python"
TORCH_HOME="${PROJECT_ROOT}/.torch"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
TASK_NAME="${TASK_NAME:-pick_place_dark_robot_raw10_quat_ep001_100_step30000}"

GPU_ID="${GPU_ID:-4}"
TARGET_STEPS="${TARGET_STEPS:-30000}"
CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-5000}"
BATCH_SIZE_TRAIN="${BATCH_SIZE_TRAIN:-16}"
BATCH_SIZE_VAL="${BATCH_SIZE_VAL:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
VALIDATION_INTERVAL_EPOCHS="${VALIDATION_INTERVAL_EPOCHS:-50}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
VAL_WORKERS="${VAL_WORKERS:-2}"
TRAIN_PREFETCH="${TRAIN_PREFETCH:-4}"
VAL_PREFETCH="${VAL_PREFETCH:-1}"
RAW_BLACK_LEVEL="${RAW_BLACK_LEVEL:-64}"
RAW_WHITE_LEVEL="${RAW_WHITE_LEVEL:-1023}"
RAW_GAMMA="${RAW_GAMMA:-2.2}"

TRAIN_RANGES="1-100"

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可用：$PYTHON_BIN" >&2; exit 1; }
[[ -d "$DATA_DIR" ]] || { echo "数据目录不存在：$DATA_DIR" >&2; exit 1; }

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NVIDIA_VISIBLE_DEVICES=all CUDA_VISIBLE_DEVICES="$GPU_ID"

CUDA_PREFLIGHT="$($PYTHON_BIN -c 'import torch; assert torch.cuda.is_available(), "CUDA 不可用"; assert torch.cuda.device_count()==1, torch.cuda.device_count(); t=torch.zeros(1,device="cuda:0"); print(f"{torch.cuda.get_device_name(0)} | torch={torch.__version__} | cuda={torch.version.cuda} | tensor_device={t.device}")')"

PREFLIGHT="$($PYTHON_BIN - "$DATA_DIR" "$BATCH_SIZE_TRAIN" "$TARGET_STEPS" <<'PY'
import glob
import h5py
import math
import os
import re
import sys

data_dir = sys.argv[1]
batch_size = int(sys.argv[2])
target_steps = int(sys.argv[3])
pattern = re.compile(r"^episode_(\d+)\.hdf5$")
episodes = {}
for path in glob.glob(os.path.join(data_dir, "*.hdf5")):
    match = pattern.match(os.path.basename(path))
    if match:
        episodes[int(match.group(1))] = path

train_ids = sorted(i for i in episodes if 1 <= i <= 100)
eval_ids = sorted(i for i in episodes if 101 <= i <= 125)
if not train_ids:
    raise SystemExit("没有找到可用训练 episode（1-100）")
if not eval_ids:
    raise SystemExit("没有找到开环评估 episode（101-125）")

lengths = []
for episode in train_ids + eval_ids:
    path = episodes[episode]
    with h5py.File(path, "r") as root:
        required = ("action", "state", "observations/qpos", "observations/images/front")
        missing = [key for key in required if key not in root]
        if missing:
            raise SystemExit(f"{path} 缺少数据集：{missing}")
        action_shape = root["action"].shape
        qpos_shape = root["observations/qpos"].shape
        state_shape = root["state"].shape
        image_shape = root["observations/images/front"].shape
        if len(action_shape) != 2 or action_shape[1] != 8:
            raise SystemExit(f"{path}: action shape={action_shape}，期待 (T,8)")
        if qpos_shape != action_shape or state_shape != action_shape:
            raise SystemExit(
                f"{path}: action={action_shape}, qpos={qpos_shape}, state={state_shape}"
            )
        if len(image_shape) != 4 or image_shape[0] != action_shape[0] or image_shape[1] != 4:
            raise SystemExit(
                f"{path}: images={image_shape}, action={action_shape}，期待 (T,4,H,W)"
            )
        encoding = root.attrs.get("image_encoding", "")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if encoding != "bggr10_planes_uint16":
            raise SystemExit(f"{path}: image_encoding={encoding!r}，期待 bggr10_planes_uint16")
        lengths.append(action_shape[0])

pool_count = len(train_ids)
inner_train_count = min(max(1, int(0.95 * pool_count)), pool_count - 1)
inner_val_count = pool_count - inner_train_count
batches_per_epoch = math.ceil(inner_train_count / batch_size)
if target_steps % batches_per_epoch:
    raise SystemExit(
        f"目标 {target_steps} steps 不能被每 epoch {batches_per_epoch} batches 整除；"
        "请调整 TARGET_STEPS 或 BATCH_SIZE_TRAIN"
    )
num_epochs = target_steps // batches_per_epoch
print(
    max(lengths), pool_count, len(eval_ids), inner_train_count,
    inner_val_count, batches_per_epoch, num_epochs
)
PY
)"

read -r EPISODE_LEN TRAIN_POOL_COUNT EVAL_COUNT INNER_TRAIN_COUNT \
  INNER_VAL_COUNT STEPS_PER_EPOCH NUM_EPOCHS <<< "$PREFLIGHT"

echo "Project:                $PROJECT_ROOT"
echo "Python:                 $PYTHON_BIN"
echo "GPU:                    $GPU_ID (单卡)"
echo "CUDA check:             $CUDA_PREFLIGHT"
echo "Dataset:                $DATA_DIR"
echo "Training ranges:        $TRAIN_RANGES"
echo "Training pool:          $TRAIN_POOL_COUNT available episodes"
echo "Internal train/val:     $INNER_TRAIN_COUNT / $INNER_VAL_COUNT episodes"
echo "Reserved open-loop:     101-125 ($EVAL_COUNT available episodes)"
echo "Episode length:         $EPISODE_LEN"
echo "State/action:           8D [x,y,z,qx,qy,qz,qw,gripper]"
echo "Image format:           RAW10 BGGR uint16 (T,4,H,W)"
echo "Train batch size:       $BATCH_SIZE_TRAIN"
echo "Steps per epoch:        $STEPS_PER_EPOCH"
echo "Epochs:                 $NUM_EPOCHS"
echo "Total optimizer steps:  $TARGET_STEPS"
echo "Checkpoint interval:    $CHECKPOINT_INTERVAL_STEPS steps"
echo "Validation interval:    $VALIDATION_INTERVAL_EPOCHS epochs"
echo "DataLoader workers:     train=$TRAIN_WORKERS val=$VAL_WORKERS"
echo "DataLoader prefetch:    train=$TRAIN_PREFETCH val=$VAL_PREFETCH"
echo "Checkpoint directory:   $CHECKPOINT_ROOT/$TASK_NAME"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1，只完成检查，不启动训练。"
  exit 0
fi

TASK_CHECKPOINT_DIR="$CHECKPOINT_ROOT/$TASK_NAME"
if [[ -d "$TASK_CHECKPOINT_DIR" ]] && \
   find "$TASK_CHECKPOINT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q . && \
   [[ "${ALLOW_EXISTING:-0}" != "1" ]]; then
  echo "Checkpoint 目录非空：$TASK_CHECKPOINT_DIR" >&2
  echo "如确认继续写入，请设置 ALLOW_EXISTING=1。" >&2
  exit 1
fi

mkdir -p "$TORCH_HOME/hub/checkpoints" "$CHECKPOINT_ROOT"
if [[ -f "$TORCH_HOME/resnet18-f37072fd.pth" && \
      ! -e "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth" ]]; then
  ln -s "$TORCH_HOME/resnet18-f37072fd.pth" \
    "$TORCH_HOME/hub/checkpoints/resnet18-f37072fd.pth"
fi

export TORCH_HOME DEVICE=cuda
export ACT_STARTOUCH_SINGLE_DATA_DIR="$DATA_DIR"
export ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR="$CHECKPOINT_ROOT"
export ACT_STARTOUCH_SINGLE_EPISODE_RANGES="$TRAIN_RANGES"
export ACT_STARTOUCH_SINGLE_EPISODE_LEN="$EPISODE_LEN"
export ACT_STARTOUCH_SINGLE_NUM_EPOCHS="$NUM_EPOCHS"
export ACT_STARTOUCH_SINGLE_TARGET_STEPS="$TARGET_STEPS"
export ACT_STARTOUCH_SINGLE_STATE_DIM=8
export ACT_STARTOUCH_SINGLE_ACTION_DIM=8
export ACT_STARTOUCH_SINGLE_BATCH_SIZE_TRAIN="$BATCH_SIZE_TRAIN"
export ACT_STARTOUCH_SINGLE_BATCH_SIZE_VAL="$BATCH_SIZE_VAL"
export ACT_STARTOUCH_SINGLE_GRAD_ACCUM="$GRAD_ACCUM"
export ACT_STARTOUCH_SINGLE_CHECKPOINT_INTERVAL_STEPS="$CHECKPOINT_INTERVAL_STEPS"
export ACT_STARTOUCH_SINGLE_VALIDATION_INTERVAL_EPOCHS="$VALIDATION_INTERVAL_EPOCHS"
export ACT_STARTOUCH_SINGLE_TRAIN_WORKERS="$TRAIN_WORKERS"
export ACT_STARTOUCH_SINGLE_VAL_WORKERS="$VAL_WORKERS"
export ACT_STARTOUCH_SINGLE_TRAIN_PREFETCH="$TRAIN_PREFETCH"
export ACT_STARTOUCH_SINGLE_VAL_PREFETCH="$VAL_PREFETCH"
export ACT_STARTOUCH_SINGLE_WANDB=0 PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
exec "$PYTHON_BIN" train.py \
  --task "$TASK_NAME" \
  --data-format raw10_hdf5 \
  --raw-black-level "$RAW_BLACK_LEVEL" \
  --raw-white-level "$RAW_WHITE_LEVEL" \
  --raw-gamma "$RAW_GAMMA"
