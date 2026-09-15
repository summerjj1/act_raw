#!/usr/bin/env python3
"""Offline/open-loop ACT evaluation on recorded Flexiv HDF5 episodes.

Recorded observations are fed to the policy; the robot is never commanded.
The script compares the first predicted action and full predicted action chunks
with recorded actions. It writes per-episode CSV metrics and a JSON summary.
"""

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


ACTION_NAMES = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint-name", default="policy_best.ckpt")
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, default=None,
                   help="Summary JSON path (default: output CSV path with .summary.json suffix).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--episode-start", type=int, default=None)
    p.add_argument("--episode-end", type=int, default=None)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--num-queries", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data-format", choices=("hdf5", "raw10_hdf5"), default="raw10_hdf5")
    p.add_argument("--raw-black-level", type=int, default=64)
    p.add_argument("--raw-white-level", type=int, default=1023)
    p.add_argument("--raw-gamma", type=float, default=2.2)
    return p.parse_args()


args = parse_args()
os.environ["ACT_STARTOUCH_SINGLE_CHECKPOINT_DIR"] = str(args.checkpoint_dir)
os.environ["ACT_STARTOUCH_SINGLE_DATA_DIR"] = str(args.data_dir)
os.environ["DEVICE"] = args.device

import h5py
import numpy as np
import torch

from config.config import SINGLE_ARM_POLICY_CONFIG
from model.utils import make_policy


def metric_summary(abs_sum, sq_sum, max_abs, count) -> dict:
    """Return aggregate metrics for 7 action dimensions and their all-dim average."""
    mae = abs_sum / max(count, 1)
    rmse = np.sqrt(sq_sum / max(count, 1))
    return {
        "count_per_dimension": int(count),
        "mae": {name: float(value) for name, value in zip(ACTION_NAMES, mae)},
        "rmse": {name: float(value) for name, value in zip(ACTION_NAMES, rmse)},
        "max_abs_error": {name: float(value) for name, value in zip(ACTION_NAMES, max_abs)},
        "mean_mae_all_dimensions": float(abs_sum.sum() / max(count * len(ACTION_NAMES), 1)),
        "rmse_all_dimensions": float(np.sqrt(sq_sum.sum() / max(count * len(ACTION_NAMES), 1))),
    }


def update(acc, error: np.ndarray) -> None:
    """Add an error tensor shaped [N, 7] to an accumulator."""
    acc["abs_sum"] += np.abs(error).sum(axis=0)
    acc["sq_sum"] += np.square(error).sum(axis=0)
    acc["max_abs"] = np.maximum(acc["max_abs"], np.abs(error).max(axis=0))
    acc["count"] += error.shape[0]


def new_accumulator() -> dict:
    return {
        "abs_sum": np.zeros(len(ACTION_NAMES), dtype=np.float64),
        "sq_sum": np.zeros(len(ACTION_NAMES), dtype=np.float64),
        "max_abs": np.zeros(len(ACTION_NAMES), dtype=np.float64),
        "count": 0,
        "gripper_correct": 0,
        "gripper_count": 0,
    }


def add_gripper_accuracy(acc, prediction: np.ndarray, target: np.ndarray) -> None:
    pred_closed = prediction[..., -1] >= 0.5
    target_closed = target[..., -1] >= 0.5
    acc["gripper_correct"] += int(np.count_nonzero(pred_closed == target_closed))
    acc["gripper_count"] += int(target_closed.size)


def gripper_accuracy(acc: dict) -> float:
    return float(acc["gripper_correct"] / max(acc["gripper_count"], 1))


def main() -> None:
    paths = sorted(args.data_dir.glob("episode_*.hdf5"))
    if args.episode_start is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) >= args.episode_start]
    if args.episode_end is not None:
        paths = [p for p in paths if int(p.stem.split("_")[-1]) <= args.episode_end]
    if args.max_episodes is not None:
        paths = paths[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"No episodes selected under {args.data_dir}")

    config = SINGLE_ARM_POLICY_CONFIG.copy()
    config["device"] = args.device
    config["num_queries"] = args.num_queries
    if args.data_format == "raw10_hdf5":
        config.update({
            "image_channels": 4,
            "image_mean": [0.406, 0.456, 0.456, 0.485],
            "image_std": [0.225, 0.224, 0.224, 0.229],
        })
    policy = make_policy(config["policy_class"], config).to(args.device)
    ckpt = args.checkpoint_dir / args.task / args.checkpoint_name
    policy.load_state_dict(torch.load(ckpt, map_location=args.device))
    policy.eval()

    stats_path = args.checkpoint_dir / args.task / "dataset_stats.pkl"
    with stats_path.open("rb") as f:
        stats = pickle.load(f)
    qmean = torch.as_tensor(stats["qpos_mean"], device=args.device, dtype=torch.float32)
    qstd = torch.as_tensor(stats["qpos_std"], device=args.device, dtype=torch.float32)
    amean = np.asarray(stats["action_mean"], dtype=np.float32)
    astd = np.asarray(stats["action_std"], dtype=np.float32)

    output_json = args.output_json or args.output_csv.with_suffix(".summary.json")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    rows, first_global, chunk_global = [], new_accumulator(), new_accumulator()

    with torch.inference_mode():
        for ep_no, path in enumerate(paths, 1):
            with h5py.File(path, "r") as h5:
                qpos = np.asarray(h5["observations/qpos"][:], dtype=np.float32)
                images = np.asarray(h5["observations/images/front"][:])
                image_encoding = h5.attrs.get("image_encoding", "rgb_uint8")
                if isinstance(image_encoding, bytes):
                    image_encoding = image_encoding.decode("utf-8")
                actions = np.asarray(h5["action"][:], dtype=np.float32)

            first_ep, chunk_ep = new_accumulator(), new_accumulator()
            for start in range(0, len(actions), args.batch_size):
                stop = min(start + args.batch_size, len(actions))
                q = (torch.from_numpy(qpos[start:stop]).to(args.device) - qmean) / qstd
                image = torch.from_numpy(images[start:stop]).to(args.device).float()
                if image_encoding == "bggr10_planes_uint16":
                    black = float(args.raw_black_level)
                    white = float(args.raw_white_level)
                    if not (0 <= black < white <= 1023):
                        raise ValueError("Invalid RAW black/white level")
                    image = torch.clamp((image - black) / (white - black), 0.0, 1.0)
                    if args.raw_gamma != 1.0:
                        image = image.pow(1.0 / args.raw_gamma)
                    image = image.unsqueeze(1)
                else:
                    image = image.permute(0, 3, 1, 2).unsqueeze(1) / 255.0
                pred_norm = policy(q, image).detach().cpu().numpy()
                pred = pred_norm * astd + amean
                for local_t in range(stop - start):
                    t = start + local_t
                    horizon = min(pred.shape[1], len(actions) - t)
                    first_error = pred[local_t, 0] - actions[t]
                    chunk_error = pred[local_t, :horizon] - actions[t:t + horizon]
                    update(first_ep, first_error[None, :])
                    update(chunk_ep, chunk_error)
                    update(first_global, first_error[None, :])
                    update(chunk_global, chunk_error)
                    add_gripper_accuracy(first_ep, pred[local_t, 0], actions[t])
                    add_gripper_accuracy(chunk_ep, pred[local_t, :horizon], actions[t:t + horizon])
                    add_gripper_accuracy(first_global, pred[local_t, 0], actions[t])
                    add_gripper_accuracy(chunk_global, pred[local_t, :horizon], actions[t:t + horizon])

            first_metrics = metric_summary(first_ep["abs_sum"], first_ep["sq_sum"], first_ep["max_abs"], first_ep["count"])
            chunk_metrics = metric_summary(chunk_ep["abs_sum"], chunk_ep["sq_sum"], chunk_ep["max_abs"], chunk_ep["count"])
            row = {
                "episode": path.name, "frames": int(len(actions)),
                "first_mae": first_metrics["mean_mae_all_dimensions"],
                "first_rmse": first_metrics["rmse_all_dimensions"],
                "chunk_mae": chunk_metrics["mean_mae_all_dimensions"],
                "chunk_rmse": chunk_metrics["rmse_all_dimensions"],
                "first_gripper_accuracy": gripper_accuracy(first_ep),
                "chunk_gripper_accuracy": gripper_accuracy(chunk_ep),
            }
            for name in ACTION_NAMES:
                row[f"first_mae_{name}"] = first_metrics["mae"][name]
                row[f"chunk_mae_{name}"] = chunk_metrics["mae"][name]
                row[f"first_rmse_{name}"] = first_metrics["rmse"][name]
                row[f"chunk_rmse_{name}"] = chunk_metrics["rmse"][name]
            rows.append(row)
            print(f"[{ep_no}/{len(paths)}] {path.name}: first_mae={row['first_mae']:.6f}, "
                  f"chunk_mae={row['chunk_mae']:.6f}, gripper_acc={row['first_gripper_accuracy']:.4f}",
                  flush=True)

    first_summary = metric_summary(first_global["abs_sum"], first_global["sq_sum"],
                                   first_global["max_abs"], first_global["count"])
    chunk_summary = metric_summary(chunk_global["abs_sum"], chunk_global["sq_sum"],
                                   chunk_global["max_abs"], chunk_global["count"])
    summary = {
        "checkpoint": str(ckpt), "dataset_stats": str(stats_path), "data_dir": str(args.data_dir),
        "episodes": [path.name for path in paths], "num_episodes": len(paths), "num_queries": args.num_queries,
        "first_action": {**first_summary, "gripper_binary_accuracy": gripper_accuracy(first_global),
                         "gripper_correct": first_global["gripper_correct"], "gripper_count": first_global["gripper_count"]},
        "full_chunk": {**chunk_summary, "gripper_binary_accuracy": gripper_accuracy(chunk_global),
                        "gripper_correct": chunk_global["gripper_correct"], "gripper_count": chunk_global["gripper_count"]},
    }

    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with output_json.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"wrote per-episode metrics: {args.output_csv}")
    print(f"wrote aggregate summary: {output_json}")
    for label, metrics in (("FIRST", summary["first_action"]), ("CHUNK", summary["full_chunk"])):
        print(f"{label} overall_mae={metrics['mean_mae_all_dimensions']:.6f} "
              f"overall_rmse={metrics['rmse_all_dimensions']:.6f} "
              f"gripper_accuracy={metrics['gripper_binary_accuracy']:.4f}")
        print("  per-dimension MAE: " + ", ".join(
            f"{name}={metrics['mae'][name]:.6f}" for name in ACTION_NAMES))


if __name__ == "__main__":
    main()
