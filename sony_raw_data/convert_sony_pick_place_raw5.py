#!/usr/bin/env python3
"""Convert Sony raw_5 BIN frames and robot H5 files for ACT.

/share/project/dy/dy1/envs/act_jack_20260824/bin/python \
/share/project/dy/dy1/code/act_raw/sony_raw_data/convert_sony_pick_place_raw5.py \
--raw-root /share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_raw_5 \
--robot-root /share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_robot_raw_5 \
--output-dir /share/project/dy/dy1/data/sony_rawdata/pick_place_sony/pick_place_sony_hdf5_raw_5

The Sony files contain two uint16 header words followed by a 6212x1940
uint16 raster.  After removing the four-pixel border, rows are interleaved
as four sensor channels (SP1 high/low and SP2 high/low).  The output keeps
the original uint16 values and stores images as (T,C,H,W).
"""
from __future__ import annotations

import argparse
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

BIN_RE = re.compile(r"_(?P<frame>\d+)\.bin$")
EP_RE = re.compile(r"episode_(?P<episode>\d+)_")
CHANNEL_NAMES = ("sp1_high", "sp1_low", "sp2_high", "sp2_low")
TIME_RE = re.compile(r"\d{8}_\d{6}")


def bin_key(path: Path) -> tuple[int, str]:
    m = BIN_RE.search(path.name)
    return (int(m.group("frame")), path.name) if m else (10**12, path.name)


def capture_time(path: Path) -> datetime:
    match = TIME_RE.search(path.name)
    if match is None:
        raise ValueError(f"采集时间无法解析: {path}")
    return datetime.strptime(match.group(), "%Y%m%d_%H%M%S")


def group_time(group: Path) -> datetime:
    first = min(group.glob("*.bin"), key=bin_key, default=None)
    if first is None:
        raise ValueError(f"没有 BIN 文件: {group}")
    return capture_time(first)


def read_bin(path: Path) -> np.ndarray:
    words = np.fromfile(path, dtype="<u2")
    if words.size < 2:
        raise ValueError(f"{path}: file is too small")
    width, height = int(words[0]), int(words[1])
    if (width, height) != (1940, 6212):
        raise ValueError(f"{path}: header says {width}x{height}, expected 1940x6212")
    if words.size != 2 + width * height:
        raise ValueError(f"{path}: {words.size} words, expected {2 + width * height}")
    raster = words[2:].reshape(height, width)[4:, 4:]
    if raster.shape != (6208, 1936) or raster.shape[0] % 4:
        raise ValueError(f"{path}: unexpected cropped raster shape {raster.shape}")
    return np.stack([raster[i::4] for i in range(4)], axis=0)


def evenly_select(length: int, count: int) -> np.ndarray:
    if count <= 0 or count > length:
        raise ValueError(f"cannot select {count} of {length}")
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def robot_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as f:
        pose = np.asarray(f["tcp_pose"], dtype=np.float64)
        grip = np.asarray(f["gripper_position"], dtype=np.float64).reshape(-1, 1)
        if pose.ndim != 2 or pose.shape[1] != 6:
            raise ValueError(f"{path}: tcp_pose must be (T,6), got {pose.shape}")
        if len(grip) != len(pose):
            raise ValueError(f"{path}: tcp_pose/gripper length mismatch")
        ts = np.asarray(f["time"], dtype=np.float64) if "time" in f else np.arange(len(pose)) / 30.0
    quat = Rotation.from_euler("xyz", pose[:, 3:6]).as_quat()
    state = np.concatenate([pose[:, :3], quat, grip], axis=1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError(f"{path}: state contains NaN/Inf")
    return state, ts


def convert_episode(group: Path, robot: Path, output: Path, args: argparse.Namespace) -> None:
    bins = sorted(group.glob("*.bin"), key=bin_key)
    if not bins:
        raise FileNotFoundError(f"no .bin files in {group}")
    state, timestamps = robot_state(robot)
    if len(bins) != len(state):
        if args.count_policy == "strict":
            raise ValueError(f"{group}: BIN={len(bins)} robot={len(state)}")
        n = min(len(bins), len(state))
        bins = [bins[int(i)] for i in evenly_select(len(bins), n)]
        idx = evenly_select(len(state), n)
        state, timestamps = state[idx], timestamps[idx]
        print(f"  frame mismatch BIN/robot; evenly trimmed to {n}")
    n = len(bins)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as out:
        action = np.vstack([state[1:], state[-1:]]) if args.action_alignment == "next" and n > 1 else state.copy()
        for key, value in (("state", state), ("action", action), ("state_xyz", state[:, :3]),
                           ("state_quaternion", state[:, 3:7]), ("state_gripper", state[:, 7:]),
                           ("observations/qpos", state)):
            out.create_dataset(key, data=value, dtype="float32", chunks=True)
        ds = out.create_dataset("observations/images/front", (n, 4, args.output_height, args.output_width),
                                dtype="uint16", chunks=(1, 4, args.output_height, args.output_width),
                                compression="lzf", shuffle=True)
        for i, path in enumerate(bins):
            frame = read_bin(path)
            ds[i] = np.stack([cv2.resize(x, (args.output_width, args.output_height), interpolation=cv2.INTER_AREA)
                              for x in frame])
            if i == 0 or i == n - 1 or (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{n}", flush=True)
        sd = h5py.string_dtype("utf-8")
        out.create_dataset("raw_frame_names", data=np.asarray([p.name for p in bins], dtype=object), dtype=sd)
        out.create_dataset("timestamps", data=timestamps, dtype="float64")
        out.attrs.update({"source_robot_h5": str(robot), "source_raw_dir": str(group),
                          "image_encoding": "sony_bin_uint16", "image_layout": "CHW",
                          "image_channel_order": ",".join(CHANNEL_NAMES), "channel_names": ",".join(CHANNEL_NAMES),
                          "raw_width": 1936, "raw_height": 1552, "output_width": args.output_width,
                          "output_height": args.output_height, "action_alignment": args.action_alignment,
                          "state_format": "xyz + quaternion_xyzw + gripper", "quaternion_order": "xyzw"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=Path, required=True); ap.add_argument("--robot-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=10**9); ap.add_argument("--output-width", type=int, default=224)
    ap.add_argument("--output-height", type=int, default=224); ap.add_argument("--count-policy", choices=("trim", "strict"), default="trim")
    ap.add_argument("--action-alignment", choices=("current", "next"), default="next"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只检查配对，不写入文件")
    args = ap.parse_args()
    if args.start < 1 or args.end < args.start:
        ap.error("需要满足 1 <= --start <= --end")
    robot_batches = sorted(
        d for d in args.robot_root.iterdir()
        if d.is_dir() and list(d.glob("episode_*.h5"))
    )
    raw_batches = sorted(
        d for d in args.raw_root.iterdir()
        if d.is_dir() and list(d.glob("index*"))
    )
    if not robot_batches or len(robot_batches) != len(raw_batches):
        raise RuntimeError(
            f"robot/raw batch 数量不一致: robot={len(robot_batches)}, raw={len(raw_batches)}"
        )
    paired = []
    for robot_batch, raw_batch in zip(robot_batches, raw_batches):
        robots = sorted(robot_batch.glob("episode_*.h5"), key=capture_time)
        groups = sorted((g for g in raw_batch.glob("index*") if g.is_dir()), key=group_time)
        if len(robots) != len(groups):
            raise RuntimeError(f"批次 {raw_batch.name}: BIN 组 {len(groups)}, 机械臂 {len(robots)}，无法安全配对")
        for group, robot in zip(groups, robots):
            delta = (group_time(group) - capture_time(robot)).total_seconds()
            if abs(delta) > 5:
                raise RuntimeError(f"时间不匹配: {group} 与 {robot} 相差 {delta:.1f} 秒")
            paired.append((group, robot, delta))
    print(f"已确认 {len(paired)} 对 Sony BIN/机械臂 episode，时间差均在 5 秒内", flush=True)
    for eid, (group, robot, delta) in enumerate(paired, start=1):
        if not args.start <= eid <= args.end:
            continue
        output = args.output_dir / f"episode_{eid:06d}.hdf5"
        if args.dry_run:
            print(f"{output.name}: {group} + {robot.name} ({delta:+.0f}s)")
            continue
        if output.exists() and not args.overwrite:
            print(f"skip existing {output}")
            continue
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".hdf5", dir=args.output_dir)
        os.close(fd)
        temp_output = Path(tmp_name)
        print(f"episode {eid}: {group} + {robot}", flush=True)
        try:
            convert_episode(group, robot, temp_output, args)
            with h5py.File(temp_output, "r+") as f:
                f.attrs["global_episode_id"] = eid
                f.attrs["source_robot_episode_id"] = int(EP_RE.search(robot.name).group("episode"))
                f.attrs["capture_time_delta_seconds"] = delta
            os.replace(temp_output, output)
        finally:
            temp_output.unlink(missing_ok=True)
        print(f"wrote {output}", flush=True)


if __name__ == "__main__": main()
