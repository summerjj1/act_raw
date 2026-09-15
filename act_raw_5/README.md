# ACT StarTouch Single

StarTouch 单臂 ACT 训练与在线推理项目。

English: Single-arm ACT training and rollout project for the StarTouch arm.

这个目录是从 `ACT_LXH` 中拆出来的单臂 StarTouch 专用版本。它移除了双臂入口和 xArm 入口，只保留：

- 单臂 HDF5 数据训练 ACT。
- StarTouch 单臂在线 rollout。
- StarTouch 专用平滑控制器 `startouch_smooth`。

English: This is a cleaned single-arm StarTouch subset of `ACT_LXH`. Dual-arm and xArm entry points are removed.

## 目录结构 / Layout

```text
ACT_StarTouch_Single/
  config/                 运行和训练配置
  detr/                   ACT / DETR 模型实现
  model/                  Policy 和数据加载工具
  scripts/
    rollout.py            StarTouch 单臂在线推理实现
    train.py              单臂训练
    train_ddp.py          单臂 DDP 训练
  checkpoints/            checkpoint 目录
  logs/                   可选 rollout 日志
  rollout.py              根目录快捷入口
  train.py                根目录快捷入口
  train_ddp.py            根目录快捷入口
  environment.yml         Conda 环境参考
```

当前已有 checkpoint 已经复制到新项目目录：

```text
checkpoints/Lumos_put_carrot_in_box
```

English: The existing checkpoint has been copied into this project.

## 环境 / Environment

使用现有环境：

```bash
conda activate dynamoe
cd /home/yan/Github/ACT_StarTouch_Single
```

设置 StarTouch SDK 路径：

```bash
export STARTOUCH_SDK_PATH="$HOME/Github/startouch_sdk/interface_py"
```

数据和 checkpoint 路径可以通过环境变量覆盖：

```bash
export ACT_STARTOUCH_SINGLE_DATA_DIR="$HOME/Github/Raw_Data/Lumos_0604_Put_Carrot_In_Box_hdf5"
export ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR="$HOME/Github/ACT_StarTouch_Single/checkpoints"
```

English: Activate the environment, export `STARTOUCH_SDK_PATH`, and optionally override dataset/checkpoint paths with environment variables.

## 启动 CAN / CAN Setup

硬件 rollout 前先启动 CAN：

```bash
cd /home/yan/Github/startouch_sdk
./start_can.sh can0 1000000
```

English: Bring up the CAN interface before running hardware rollout.

## 在线推理 / Rollout

推荐默认使用 StarTouch 专用控制器：

```bash
cd /home/yan/Github/ACT_StarTouch_Single

STARTOUCH_SDK_PATH="$HOME/Github/startouch_sdk/interface_py" \
python rollout.py \
  --task Lumos_put_carrot_in_box \
  --can can0 \
  --camera_dev 0 \
  --ckpt_name policy_last.ckpt \
  --init_pose 0.4,0.0,0.16,0.0,0.0,0.0 \
  --tcp_offset 0.0,0.0,0.0 \
  --num_action_steps 40 \
  --skip_action_steps 10 \
  --controller_mode startouch_smooth \
  --waypoint_control_hz 15 \
  --waypoint_time_scale 1.0 \
  --rolling_switch_delay 0.05 \
  --rolling_max_retries 3 \
  --rolling_max_time_scale 5 \
  --gripper_close_gain 1.2
```

English: `startouch_smooth` is the recommended default controller for StarTouch online rollout.

### 常用调参 / Common Tuning

更平滑，但会更慢：

```bash
--startouch_goal_alpha 0.5 --startouch_smooth_window 7 --waypoint_time_scale 1.3
```

更快，但可能更接近速度限制：

```bash
--waypoint_control_hz 20 --waypoint_time_scale 0.8
```

保留 chunk 中间的稀疏点，而不是只走首尾点：

```bash
--startouch_chunk_mode stride --startouch_stride 5
```

English: Increase smoothing and time scale for stability; increase control Hz or reduce time scale for speed; use stride mode to preserve sparse intermediate keyframes.

## 控制器模式 / Controller Modes

`raw`

直接 TCP raw servo，带轻量 action 滤波。速度快，但对模型输出抖动非常敏感。

English: Direct TCP raw servo. Fast, but sensitive to model jitter.

`waypoint`

阻塞式 SDK waypoint 执行。轨迹更平滑，但 Python 会等当前 chunk 执行完，再继续读下一帧相机和推理。

English: Blocking SDK waypoint execution. Smooth, but delays the next camera read and policy inference.

`rolling_waypoint`

非阻塞 SDK waypoint update。主线程持续读相机和推理，后台线程更新未来轨迹。

English: Non-blocking waypoint update. The main thread keeps sensing and inferring while a background thread updates the future trajectory.

`startouch_smooth`

StarTouch 专用模式，推荐默认使用。它会：

- 平滑模型输出的 action chunk。
- 选择首尾点或稀疏关键点。
- 在 chunk 前插入当前机械臂状态，避免第一步跳变。
- 对 TCP/RPY 间隔过大的关键点做 densify。
- 使用 rolling waypoint update。
- 遇到 SDK 关节速度超限时自动拉长 `time_sec` 重试。

English: StarTouch-specific controller. It smooths chunks, selects keyframes, prepends current state, densifies large gaps, uses rolling waypoint update, and retries with slower timing on SDK velocity-limit errors.

## 训练 / Training

单卡训练：

```bash
cd /home/yan/Github/ACT_StarTouch_Single

ACT_STARTOUCH_SINGLE_DATA_DIR="$HOME/Github/Raw_Data/Lumos_0604_Put_Carrot_In_Box_hdf5" \
ACT_STARTOUCH_SINGLE_WANDB=0 \
python train.py --task Lumos_put_carrot_in_box
```

DDP 多卡训练：

```bash
torchrun --nproc_per_node=4 train_ddp.py --task Lumos_put_carrot_in_box
```

English: Use `train.py` for single-process training and `train_ddp.py` with `torchrun` for distributed training.

## 配置 / Configuration

主要配置在：

[config/config.py](config/config.py)

也可以用环境变量覆盖：

```bash
ACT_STARTOUCH_SINGLE_DATA_DIR
ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR
ACT_STARTOUCH_SINGLE_EPISODE_LEN
ACT_STARTOUCH_SINGLE_NUM_EPOCHS
ACT_STARTOUCH_SINGLE_BATCH_SIZE_TRAIN
ACT_STARTOUCH_SINGLE_BATCH_SIZE_VAL
ACT_STARTOUCH_SINGLE_GRAD_ACCUM
ACT_STARTOUCH_SINGLE_EVAL_CKPT
ACT_STARTOUCH_SINGLE_WANDB
```

English: Edit `config/config.py` or override settings with environment variables.

## 说明 / Notes

这个项目只保留 StarTouch 单臂代码。双臂 rollout、xArm rollout，以及当前 `ACT_LXH` 中已经缺失的 `tool/` 数据处理工具没有放进来。

English: This project intentionally keeps only StarTouch single-arm code. Dual-arm rollout, xArm rollout, and unavailable `ACT_LXH/tool` utilities are not included.
