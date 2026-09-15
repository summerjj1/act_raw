import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_ROOT = Path.home() / "Github" / "Raw_Data"
USE_DEPTH_ANYTHING = False
USE_LANG_SAM = False


def _resolve_path(env_key, default_path):
    raw_value = os.environ.get(env_key)
    if raw_value:
        return str(Path(raw_value).expanduser().resolve())
    return str(Path(default_path).expanduser().resolve())


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["DEVICE"] = DEVICE

DATA_DIR = _resolve_path(
    "ACT_STARTOUCH_SINGLE_DATA_DIR",
    RAW_DATA_ROOT / "Lumos_0604_Put_Carrot_In_Box_hdf5",
)

STATE_DIM = int(os.environ.get("ACT_STARTOUCH_SINGLE_STATE_DIM", "7"))
ACTION_DIM = int(os.environ.get("ACT_STARTOUCH_SINGLE_ACTION_DIM", str(STATE_DIM)))
CHECKPOINT_DIR = _resolve_path(
    "ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR",
    PROJECT_ROOT / "checkpoints",
)


SINGLE_ARM_TASK_CONFIG = {
    "dataset_dir": DATA_DIR,
    "episode_len": int(os.environ.get("ACT_STARTOUCH_SINGLE_EPISODE_LEN", "399")),
    "state_dim": STATE_DIM,
    "action_dim": ACTION_DIM,
    "cam_width": 224,
    "cam_height": 224,
    "camera_names": ["front"],
    "episode_ranges": os.environ.get("ACT_STARTOUCH_SINGLE_EPISODE_RANGES") or None,
}


SINGLE_ARM_POLICY_CONFIG = {
    "lr": 1e-5,
    "device": DEVICE,
    "num_queries": 80,
    "kl_weight": 100,
    "hidden_dim": 512,
    "dim_feedforward": 3200,
    "lr_backbone": 1e-5,
    "backbone": "resnet18",
    "enc_layers": 4,
    "dec_layers": 7,
    "nheads": 8,
    "camera_names": ["front"],
    "policy_class": "ACT",
    "temporal_agg": False,
    "state_dim": STATE_DIM,
    "detrvae_input_dim": STATE_DIM,
}


SINGLE_ARM_TRAIN_CONFIG = {
    "seed": 42,
    "num_epochs": int(os.environ.get("ACT_STARTOUCH_SINGLE_NUM_EPOCHS", "1000")),
    "batch_size_train": int(os.environ.get("ACT_STARTOUCH_SINGLE_BATCH_SIZE_TRAIN", "16")),
    "batch_size_val": int(os.environ.get("ACT_STARTOUCH_SINGLE_BATCH_SIZE_VAL", "16")),
    "grad_accum": int(os.environ.get("ACT_STARTOUCH_SINGLE_GRAD_ACCUM", "1")),
    "checkpoint_interval_steps": int(os.environ.get("ACT_STARTOUCH_SINGLE_CHECKPOINT_INTERVAL_STEPS", "4000")),
    "eval_ckpt_name": os.environ.get("ACT_STARTOUCH_SINGLE_EVAL_CKPT", "policy_last.ckpt"),
    "checkpoint_dir": CHECKPOINT_DIR,
}


WANDB_CONFIG = {
    "enabled": os.environ.get("ACT_STARTOUCH_SINGLE_WANDB", "1") not in {"0", "false", "False"},
    "single_arm_project": os.environ.get("ACT_STARTOUCH_SINGLE_WANDB_PROJECT", "ACT_StarTouch_Single"),
}
