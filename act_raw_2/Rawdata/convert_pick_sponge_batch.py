#!/usr/bin/env python3
"""Batch-convert grouped pick-sponge NV12 data and matching robot H5 episodes.

Pairing is explicit by index: group_0007 is paired with robot file containing
_ep7_.  Each pair creates episode_000007.hdf5 and episode_000007.mp4.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import h5py

EP_RE = re.compile(r"_ep(?P<episode>\d+)_")
GROUP_RE = re.compile(r"^group_(?P<episode>\d+)$")


def episode_from_robot_path(path: Path) -> int:
    match = EP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse _epN_ from robot file: {path.name}")
    return int(match["episode"])


def episode_from_group_path(path: Path) -> int:
    match = GROUP_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected group directory name: {path.name}")
    return int(match["episode"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="directory for MP4 files (default: OUTPUT_DIR/videos)",
    )
    parser.add_argument("--converter", type=Path,
                        default=Path(__file__).with_name("convert_pick_sponge_h5.py"))
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--robot-episode-offset",
        type=int,
        default=0,
        help="robot episode paired with group_N is N + this offset (default: 0)",
    )
    parser.add_argument(
        "--output-episode-offset",
        type=int,
        default=None,
        help="output episode number is group_N + this offset (default: robot offset)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir or (args.output_dir / "videos")
    groups = {
        episode_from_group_path(path): path
        for path in args.group_root.glob("group_*")
        if path.is_dir()
    }
    robots = {
        episode_from_robot_path(path): path
        for path in [*args.robot_dir.glob("*.h5"), *args.robot_dir.glob("*.hdf5")]
    }
    if not groups:
        raise RuntimeError(f"No group_#### directories under {args.group_root}")
    if not robots:
        raise RuntimeError(f"No robot .h5/.hdf5 files under {args.robot_dir}")
    if not args.converter.is_file():
        raise FileNotFoundError(args.converter)
    robot_indices = {
        group_index + args.robot_episode_offset
        for group_index in groups
    }
    if not robot_indices.issubset(robots):
        raise RuntimeError(
            f"Missing robot episodes for groups: {sorted(robot_indices - set(robots))}; "
            f"available robot episodes={sorted(robots)}"
        )

    output_offset = (
        args.robot_episode_offset
        if args.output_episode_offset is None
        else args.output_episode_offset
    )
    plan = []
    for group_episode in sorted(groups):
        robot_episode = group_episode + args.robot_episode_offset
        output_episode = group_episode + output_offset
        group = groups[group_episode]
        robot = robots[robot_episode]
        nv12_count = len(list((group / "nv12").glob("*.yuv")))
        with h5py.File(robot, "r") as handle:
            robot_count = int(handle["timestamps"].shape[0])
        if nv12_count == 0:
            raise RuntimeError(f"No NV12 frames: {group}")
        plan.append(
            (group_episode, robot_episode, output_episode, group, robot, nv12_count, robot_count)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "conversion_manifest.csv"
    print(
        f"preflight OK: pairs={len(plan)} exact="
        f"{sum(nv12 == robot for *_, nv12, robot in plan)} "
        f"mismatched={sum(nv12 != robot for *_, nv12, robot in plan)}"
    )
    for group_episode, robot_episode, output_episode, _, _, nv12_count, robot_count in plan:
        print(
            f"group={group_episode:04d} robot_ep={robot_episode:04d} "
            f"output_ep={output_episode:04d} nv12={nv12_count} robot={robot_count}"
        )

    if args.dry_run:
        return

    rows = []
    for offset, (
        group_episode, robot_episode, output_episode, group, robot, nv12_count, robot_count
    ) in enumerate(plan, start=1):
        hdf5_path = args.output_dir / f"episode_{output_episode:06d}.hdf5"
        mp4_path = video_dir / f"episode_{output_episode:06d}.mp4"
        if hdf5_path.exists() and mp4_path.exists() and not args.overwrite:
            print(f"[{offset}/{len(plan)}] skip existing output episode={output_episode:04d}")
        else:
            command = [
                sys.executable, str(args.converter),
                "--group-dir", str(group),
                "--robot-h5", str(robot),
                "--output", str(hdf5_path),
                "--video-output", str(mp4_path),
                "--video-fps", str(args.video_fps),
                "--count-policy", "trim",
            ]
            if args.overwrite:
                command.append("--overwrite")
            print(
                f"[{offset}/{len(plan)}] convert group={group_episode:04d} "
                f"robot_ep={robot_episode:04d} output_ep={output_episode:04d}",
                flush=True,
            )
            subprocess.run(command, check=True)
        rows.append([
            group_episode, robot_episode, output_episode, group, robot, nv12_count, robot_count,
            min(nv12_count, robot_count), hdf5_path, mp4_path,
        ])

    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "group_episode", "robot_episode", "output_episode", "group_dir", "robot_h5", "nv12_frames", "robot_frames",
            "output_frames", "hdf5", "video",
        ])
        writer.writerows(rows)
    print(f"done: {len(rows)} episodes; manifest={manifest_path}")


if __name__ == "__main__":
    main()
