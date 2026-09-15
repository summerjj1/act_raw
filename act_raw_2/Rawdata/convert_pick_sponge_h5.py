#!/usr/bin/env python3
"""Convert one grouped NV12 episode plus its robot H5 log to flexiv_act HDF5.

The output matches flexiv_act/bash_tran_rawdata/train_pick_sponge.sh:
  /action
  /observations/qpos
  /observations/images/front

State is the current TCP pose in xyz + rpy + binary gripper-command format.
With --action-alignment next (the default), action[t] is state[t+1] and
the final action repeats the final state.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

TIMESTAMP_RE = re.compile(r"(?P<date>\d{8})_(?P<hms>\d{6})_(?P<ms>\d{3})_")


def frame_key(path: Path) -> tuple[int, int, int, str]:
    match = TIMESTAMP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse timestamp from {path.name}")
    return (
        int(match["date"]),
        int(match["hms"]),
        int(match["ms"]),
        path.name,
    )


def read_i420(path: Path, width: int, height: int, output_size: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    expected = width * height * 3 // 2
    if raw.size != expected:
        raise ValueError(
            f"{path} has {raw.size} bytes; expected {expected} for "
            f"{width}x{height} YUV420"
        )
    yuv = raw.reshape((height * 3 // 2, width))
    # Camera files are explicitly named *_NV12.yuv: Y plane followed by
    # interleaved UV chroma (not planar I420).
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(
        rgb, (output_size, output_size), interpolation=cv2.INTER_AREA
    )


def build_states(handle: h5py.File) -> np.ndarray:
    pose = np.asarray(handle["tcp_pose"][:], dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"tcp_pose must have shape (T, 7), got {pose.shape}")
    quat = pose[:, 3:7]
    norms = np.linalg.norm(quat, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-8):
        raise ValueError("tcp_pose contains invalid quaternions")
    quat = quat / norms
    rpy = Rotation.from_quat(quat).as_euler("xyz", degrees=False)

    if "gripper_command" not in handle:
        raise KeyError("H5 must contain gripper_command")
    gripper = np.asarray(handle["gripper_command"][:], dtype=np.float64)
    gripper = (gripper > 0).astype(np.float64).reshape(-1, 1)
    states = np.concatenate([pose[:, :3], rpy, gripper], axis=1)
    if not np.all(np.isfinite(states)):
        raise ValueError("state contains NaN or Inf")
    return states.astype(np.float32)


def build_actions(states: np.ndarray, alignment: str) -> np.ndarray:
    actions = states.copy()
    if alignment == "next" and len(states) > 1:
        actions[:-1] = states[1:]
    return actions


def evenly_select(values, count: int):
    """Keep `count` temporally even samples, including first and last."""
    if count <= 0 or count > len(values):
        raise ValueError(f"Cannot select {count} values from {len(values)}")
    if count == len(values):
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(np.int64)
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(index)] for index in indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-dir", type=Path, required=True)
    parser.add_argument("--robot-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="optional MP4 path; contains the same resized RGB frames written to HDF5",
    )
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--action-alignment",
        choices=("current", "next"),
        default="next",
        help="current: action[t]=state[t]; next: action[t]=state[t+1]",
    )
    parser.add_argument(
        "--count-policy",
        choices=("strict", "trim"),
        default="strict",
        help=("strict requires equal NV12/robot frame counts; trim evenly removes "
              "extra frames from the longer stream"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nv12_dir = args.group_dir / "nv12"
    raw10_dir = args.group_dir / "raw10"
    nv12_files = sorted(nv12_dir.glob("*.yuv"), key=frame_key)
    raw10_files = sorted(raw10_dir.glob("*.raw"), key=frame_key)
    if not nv12_files:
        raise SystemExit(f"No .yuv files found under {nv12_dir}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {args.output}")
    if args.video_output is not None and args.video_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Video output exists; pass --overwrite: {args.video_output}"
        )
    if args.video_fps <= 0:
        raise ValueError("--video-fps must be positive")

    with h5py.File(args.robot_h5, "r") as robot:
        required = ("timestamps", "tcp_pose")
        missing = [key for key in required if key not in robot]
        if missing:
            raise KeyError(f"Missing datasets in robot H5: {missing}")
        states = build_states(robot)
        robot_timestamps = np.asarray(robot["timestamps"][:], dtype=np.float64)
        if len(states) != len(nv12_files):
            if args.count_policy == "strict":
                raise ValueError(
                    f"Frame mismatch: NV12={len(nv12_files)}, robot states={len(states)}"
                )
            common_count = min(len(nv12_files), len(states))
            print(
                f"frame mismatch: NV12={len(nv12_files)}, robot={len(states)}; "
                f"evenly trimming both to {common_count}",
                flush=True,
            )
            nv12_files = evenly_select(nv12_files, common_count)
            states = evenly_select(states, common_count)
            robot_timestamps = evenly_select(robot_timestamps, common_count)
        if raw10_files and len(raw10_files) != len(nv12_files):
            warnings.warn(
                f"RAW10 count ({len(raw10_files)}) differs from NV12 count "
                f"({len(nv12_files)}); RAW10 is not stored by flexiv_act."
            )
        actions = build_actions(states, args.action_alignment)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(args.output, "w") as out:
            out.create_dataset(
                "action",
                data=actions,
                dtype="float32",
                chunks=True,
            )
            observations = out.create_group("observations")
            observations.create_dataset(
                "qpos",
                data=states,
                dtype="float32",
                chunks=True,
            )
            image_ds = observations.create_dataset(
                "images/front",
                shape=(len(nv12_files), args.output_size, args.output_size, 3),
                dtype="uint8",
                chunks=(1, args.output_size, args.output_size, 3),
                compression="lzf",
                shuffle=True,
            )
            video_writer = None
            if args.video_output is not None:
                args.video_output.parent.mkdir(parents=True, exist_ok=True)
                video_writer = cv2.VideoWriter(
                    str(args.video_output),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    args.video_fps,
                    (args.output_size, args.output_size),
                )
                if not video_writer.isOpened():
                    raise RuntimeError(f"Cannot create MP4: {args.video_output}")
            try:
                for index, path in enumerate(nv12_files):
                    frame_rgb = read_i420(
                        path, args.width, args.height, args.output_size
                    )
                    image_ds[index] = frame_rgb
                    if video_writer is not None:
                        video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                    if index % 100 == 0 or index == len(nv12_files) - 1:
                        print(f"converted {index + 1}/{len(nv12_files)} frames", flush=True)
            finally:
                if video_writer is not None:
                    video_writer.release()

            out.attrs["source_group_dir"] = str(args.group_dir)
            out.attrs["source_robot_h5"] = str(args.robot_h5)
            out.attrs["action_alignment"] = args.action_alignment
            out.attrs["state_format"] = "tcp_pose xyz + rpy_xyz + gripper_open"
            out.attrs["gripper_source"] = "gripper_command (binary 0/1)"
            out.attrs["image_format"] = "NV12/YUV420 -> RGB, resized to output_size"
            out.create_dataset("timestamps", data=robot_timestamps, dtype="float64")

    print(f"wrote {args.output}")
    print(f"frames={len(nv12_files)} state_shape={states.shape} action_shape={actions.shape}")
    print(f"action_alignment={args.action_alignment}")
    if args.video_output is not None:
        print(f"video={args.video_output} fps={args.video_fps:g}")


if __name__ == "__main__":
    main()
