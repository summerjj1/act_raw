#!/usr/bin/env python3
"""Open-loop evaluation for RAW10 ACT policies with 8D quaternion actions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ACTION_NAMES = ("x", "y", "z", "qx", "qy", "qz", "qw", "gripper")
QUAT_SLICE = slice(3, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint-name", default="policy_best.ckpt")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit episode IDs to evaluate, for example: 10 20 30",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--raw-black-level", type=int, default=64)
    parser.add_argument("--raw-white-level", type=int, default=1023)
    parser.add_argument("--raw-gamma", type=float, default=2.2)
    return parser.parse_args()


args = parse_args()
os.environ["ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR"] = str(args.checkpoint_dir)
os.environ["ACT_STARTOUCH_SINGLE_DATA_DIR"] = str(args.data_dir)
os.environ["ACT_STARTOUCH_SINGLE_STATE_DIM"] = "8"
os.environ["ACT_STARTOUCH_SINGLE_ACTION_DIM"] = "8"
os.environ["DEVICE"] = args.device

import h5py
import numpy as np
import torch

from config.config import SINGLE_ARM_POLICY_CONFIG
from model.utils import make_policy


def episode_number(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    return quaternion / np.maximum(norm, 1e-12)


def quaternion_aligned_error(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sign-aligned 8D component error and SO(3) angle error.

    q and -q represent the same rotation. Predictions are normalized and their
    signs aligned to the normalized target before component errors are taken.
    Rotation error is the geodesic angle 2*acos(abs(dot(q_pred, q_target))).
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or prediction.shape[-1] != 8:
        raise ValueError(
            f"Expected matching (...,8) arrays, got {prediction.shape} and {target.shape}"
        )

    pred_quat = normalize_quaternion(prediction[..., QUAT_SLICE])
    target_quat = normalize_quaternion(target[..., QUAT_SLICE])
    signed_dot = np.sum(pred_quat * target_quat, axis=-1, keepdims=True)
    aligned_pred_quat = np.where(signed_dot < 0.0, -pred_quat, pred_quat)

    aligned_prediction = prediction.copy()
    aligned_target = target.copy()
    aligned_prediction[..., QUAT_SLICE] = aligned_pred_quat
    aligned_target[..., QUAT_SLICE] = target_quat
    component_error = aligned_prediction - aligned_target

    dot = np.clip(np.abs(np.sum(pred_quat * target_quat, axis=-1)), 0.0, 1.0)
    angle_error = 2.0 * np.arccos(dot)
    return component_error, angle_error


def new_accumulator() -> dict:
    return {
        "abs_sum": np.zeros(8, dtype=np.float64),
        "sq_sum": np.zeros(8, dtype=np.float64),
        "max_abs": np.zeros(8, dtype=np.float64),
        "count": 0,
        "angle_abs_sum": 0.0,
        "angle_sq_sum": 0.0,
        "angle_max": 0.0,
    }


def update(accumulator: dict, prediction: np.ndarray, target: np.ndarray) -> None:
    prediction = np.asarray(prediction).reshape(-1, 8)
    target = np.asarray(target).reshape(-1, 8)
    component_error, angle_error = quaternion_aligned_error(prediction, target)
    absolute = np.abs(component_error)
    accumulator["abs_sum"] += absolute.sum(axis=0)
    accumulator["sq_sum"] += np.square(component_error).sum(axis=0)
    accumulator["max_abs"] = np.maximum(accumulator["max_abs"], absolute.max(axis=0))
    accumulator["count"] += component_error.shape[0]
    accumulator["angle_abs_sum"] += float(np.abs(angle_error).sum())
    accumulator["angle_sq_sum"] += float(np.square(angle_error).sum())
    accumulator["angle_max"] = max(
        accumulator["angle_max"], float(np.max(np.abs(angle_error)))
    )


def summarize(accumulator: dict) -> dict:
    count = max(int(accumulator["count"]), 1)
    mae = accumulator["abs_sum"] / count
    rmse = np.sqrt(accumulator["sq_sum"] / count)
    angle_mae = accumulator["angle_abs_sum"] / count
    angle_rmse = np.sqrt(accumulator["angle_sq_sum"] / count)
    return {
        "count": int(accumulator["count"]),
        "mae": dict(zip(ACTION_NAMES, map(float, mae))),
        "rmse": dict(zip(ACTION_NAMES, map(float, rmse))),
        "max_abs_error": dict(zip(ACTION_NAMES, map(float, accumulator["max_abs"]))),
        "xyz_mae": float(mae[:3].mean()),
        "quaternion_component_mae": float(mae[3:7].mean()),
        "gripper_mae": float(mae[7]),
        "rotation_angle_mae_rad": float(angle_mae),
        "rotation_angle_rmse_rad": float(angle_rmse),
        "rotation_angle_max_rad": float(accumulator["angle_max"]),
        "rotation_angle_mae_deg": float(np.degrees(angle_mae)),
        "rotation_angle_rmse_deg": float(np.degrees(angle_rmse)),
        "rotation_angle_max_deg": float(np.degrees(accumulator["angle_max"])),
    }


def main() -> None:
    paths = sorted(args.data_dir.glob("episode_*.hdf5"), key=episode_number)
    if args.episode_ids is not None:
        if args.episode_start is not None or args.episode_end is not None:
            raise ValueError(
                "--episode-ids cannot be combined with --episode-start/--episode-end"
            )
        requested = set(args.episode_ids)
        if len(requested) != len(args.episode_ids):
            raise ValueError("--episode-ids contains duplicate values")
        available = {episode_number(path) for path in paths}
        missing = sorted(requested - available)
        if missing:
            raise FileNotFoundError(
                f"Requested episodes are missing under {args.data_dir}: {missing}"
            )
        paths = [path for path in paths if episode_number(path) in requested]
    if args.episode_start is not None:
        paths = [path for path in paths if episode_number(path) >= args.episode_start]
    if args.episode_end is not None:
        paths = [path for path in paths if episode_number(path) <= args.episode_end]
    if args.max_episodes is not None:
        paths = paths[:args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"No episodes selected under {args.data_dir}")

    checkpoint = args.checkpoint_dir / args.task / args.checkpoint_name
    stats_path = args.checkpoint_dir / args.task / "dataset_stats.pkl"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)

    config = SINGLE_ARM_POLICY_CONFIG.copy()
    config.update({"device": args.device, "num_queries": args.num_queries,
                   "state_dim": 8, "detrvae_input_dim": 8,
                   "image_channels": 4,
                   "image_mean": [0.0, 0.0, 0.0, 0.0],
                   "image_std": [1.0, 1.0, 1.0, 1.0]})
    policy = make_policy(config["policy_class"], config).to(args.device)
    policy.load_state_dict(torch.load(checkpoint, map_location=args.device))
    policy.eval()

    with stats_path.open("rb") as handle:
        stats = pickle.load(handle)
    for name in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
        if np.asarray(stats[name]).shape != (8,):
            raise ValueError(f"{stats_path}: {name} shape={np.asarray(stats[name]).shape}, expected (8,)")
    qmean = torch.as_tensor(stats["qpos_mean"], device=args.device, dtype=torch.float32)
    qstd = torch.as_tensor(stats["qpos_std"], device=args.device, dtype=torch.float32)
    action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
    action_std = np.asarray(stats["action_std"], dtype=np.float32)

    output_json = args.output_json or args.output_csv.with_suffix(".summary.json")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    first_global = new_accumulator()
    chunk_global = new_accumulator()
    rows = []

    with torch.inference_mode():
        for index, path in enumerate(paths, 1):
            with h5py.File(path, "r") as h5:
                qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float32)
                images = np.asarray(h5["observations/images/front"][:])
                actions = np.asarray(h5["action"][:], dtype=np.float32)
                image_encoding = h5.attrs.get("image_encoding", "")
                if isinstance(image_encoding, bytes):
                    image_encoding = image_encoding.decode("utf-8")
            if qpos.shape[-1] != 8 or actions.shape[-1] != 8:
                raise ValueError(f"{path}: qpos={qpos.shape}, action={actions.shape}; expected 8D")
            if images.ndim != 4 or images.shape[1] != 4:
                raise ValueError(f"{path}: expected RAW10 (T,4,H,W), got {images.shape}")
            if image_encoding not in ("bggr10_planes_uint16", "sony_bin_uint16"):
                raise ValueError(
                    f"{path}: expected Sony or RAW10 image encoding, "
                    f"got {image_encoding!r}"
                )
            if not (0 <= args.raw_black_level < args.raw_white_level <= 65535):
                raise ValueError("Invalid RAW/Sony black/white level")

            first_episode = new_accumulator()
            chunk_episode = new_accumulator()
            for start in range(0, len(actions), args.batch_size):
                stop = min(start + args.batch_size, len(actions))
                qpos_batch = torch.from_numpy(qpos[start:stop]).to(args.device)
                qpos_batch = (qpos_batch - qmean) / qstd
                image_batch = torch.from_numpy(images[start:stop]).to(args.device).float()
                image_batch = torch.clamp(
                    (image_batch - float(args.raw_black_level))
                    / float(args.raw_white_level - args.raw_black_level),
                    0.0,
                    1.0,
                )
                if args.raw_gamma != 1.0:
                    image_batch = image_batch.pow(1.0 / args.raw_gamma)
                image_batch = image_batch.unsqueeze(1)
                prediction_normalized = policy(qpos_batch, image_batch).detach().cpu().numpy()
                prediction = prediction_normalized * action_std + action_mean

                for local_index in range(stop - start):
                    timestep = start + local_index
                    horizon = min(prediction.shape[1], len(actions) - timestep)
                    first_prediction = prediction[local_index, 0:1]
                    first_target = actions[timestep:timestep + 1]
                    chunk_prediction = prediction[local_index, :horizon]
                    chunk_target = actions[timestep:timestep + horizon]
                    update(first_episode, first_prediction, first_target)
                    update(chunk_episode, chunk_prediction, chunk_target)
                    update(first_global, first_prediction, first_target)
                    update(chunk_global, chunk_prediction, chunk_target)

            first = summarize(first_episode)
            chunk = summarize(chunk_episode)
            row = {
                "episode": path.name,
                "frames": len(actions),
                "first_xyz_mae": first["xyz_mae"],
                "first_rotation_angle_mae_deg": first["rotation_angle_mae_deg"],
                "first_quaternion_component_mae": first["quaternion_component_mae"],
                "first_gripper_mae": first["gripper_mae"],
                "chunk_xyz_mae": chunk["xyz_mae"],
                "chunk_rotation_angle_mae_deg": chunk["rotation_angle_mae_deg"],
                "chunk_quaternion_component_mae": chunk["quaternion_component_mae"],
                "chunk_gripper_mae": chunk["gripper_mae"],
            }
            for name in ACTION_NAMES:
                row[f"first_mae_{name}"] = first["mae"][name]
                row[f"chunk_mae_{name}"] = chunk["mae"][name]
            rows.append(row)
            print(
                f"[{index}/{len(paths)}] {path.name}: "
                f"first_xyz={first['xyz_mae']:.6f}, "
                f"first_rot={first['rotation_angle_mae_deg']:.3f} deg, "
                f"first_quat=[qx={first['mae']['qx']:.6f}, "
                f"qy={first['mae']['qy']:.6f}, qz={first['mae']['qz']:.6f}, "
                f"qw={first['mae']['qw']:.6f}], "
                f"first_gripper={first['gripper_mae']:.6f}",
                flush=True,
            )

    summary = {
        "checkpoint": str(checkpoint),
        "dataset_stats": str(stats_path),
        "data_dir": str(args.data_dir),
        "quaternion_order": "xyzw",
        "image_encoding": "sony_bin_uint16",
        "raw_black_level": args.raw_black_level,
        "raw_white_level": args.raw_white_level,
        "raw_gamma": args.raw_gamma,
        "quaternion_metric_note": (
            "Prediction and target are unit-normalized; prediction sign is aligned "
            "because q and -q represent the same rotation. Rotation angle uses "
            "2*acos(abs(dot(q_pred,q_target)))."
        ),
        "episodes": [path.name for path in paths],
        "num_episodes": len(paths),
        "num_queries": args.num_queries,
        "first_action": summarize(first_global),
        "full_chunk": summarize(chunk_global),
    }
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with output_json.open("w") as handle:
        json.dump(summary, handle, indent=2)

    markdown = args.output_markdown or args.output_csv.with_suffix(".md")
    first, chunk = summary["first_action"], summary["full_chunk"]
    markdown.write_text(
        "# Sony 开环测试结果\n\n"
        f"- Checkpoint: `{checkpoint}`\n"
        f"- 测试集: episode {episode_number(paths[0])}–{episode_number(paths[-1])}（{len(paths)} 集）\n"
        f"- 图像编码: `{summary['image_encoding']}`，黑白电平 `{args.raw_black_level}/{args.raw_white_level}`，gamma `{args.raw_gamma}`\n"
        f"- 查询长度: {args.num_queries}\n\n"
        "## 聚合指标\n\n"
        "| 模式 | XYZ MAE | 旋转角 MAE (deg) | 四元数分量 MAE | Gripper MAE |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| First action | {first['xyz_mae']:.6f} | {first['rotation_angle_mae_deg']:.3f} | {first['quaternion_component_mae']:.6f} | {first['gripper_mae']:.6f} |\n"
        f"| Full chunk | {chunk['xyz_mae']:.6f} | {chunk['rotation_angle_mae_deg']:.3f} | {chunk['quaternion_component_mae']:.6f} | {chunk['gripper_mae']:.6f} |\n\n"
        "逐 episode 指标见同目录 CSV；完整统计见 JSON。\n"
    )

    print(f"wrote per-episode metrics: {args.output_csv}")
    print(f"wrote aggregate summary: {output_json}")
    for label, metrics in (("FIRST", summary["first_action"]), ("CHUNK", summary["full_chunk"])):
        print(
            f"{label}: xyz_mae={metrics['xyz_mae']:.6f}, "
            f"rotation_mae={metrics['rotation_angle_mae_deg']:.3f} deg, "
            f"quat_component_mae={metrics['quaternion_component_mae']:.6f}, "
            f"gripper_mae={metrics['gripper_mae']:.6f}"
        )
        print(
            f"  quaternion component MAE: qx={metrics['mae']['qx']:.6f}, "
            f"qy={metrics['mae']['qy']:.6f}, qz={metrics['mae']['qz']:.6f}, "
            f"qw={metrics['mae']['qw']:.6f}"
        )


if __name__ == "__main__":
    main()
