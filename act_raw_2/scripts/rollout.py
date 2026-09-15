#!/usr/bin/env python3
# ACT single-arm rollout for Startouch.
# ACT action/state convention: xyz + rpy(rad) + gripper_open, all positions in meters.

import argparse
import faulthandler
import json
import os
import pickle
import queue
import re
import sys
import threading
import time

import cv2
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DETR_PATH = os.path.join(PROJECT_ROOT, 'detr')
sys.path.append(DETR_PATH)

from config.config import SINGLE_ARM_TASK_CONFIG, SINGLE_ARM_POLICY_CONFIG, SINGLE_ARM_TRAIN_CONFIG
from model.utils import make_policy


faulthandler.enable()


def log_probe(message):
    print(message, flush=True)


def _to_float_list(arr):
    return np.asarray(arr, dtype=np.float64).round(8).tolist()


def _trajectory_limit_time_multiplier(error_message):
    ratios = []
    for actual, limit in re.findall(r'vmax\s+([0-9.eE+-]+)/([0-9.eE+-]+)', error_message):
        actual_f = float(actual)
        limit_f = float(limit)
        if limit_f > 0.0 and actual_f > limit_f:
            ratios.append(actual_f / limit_f)
    return max(ratios) if ratios else None


def _moving_average_actions(actions, window):
    actions = np.asarray(actions, dtype=np.float64)
    window = int(window)
    if window <= 1 or len(actions) <= 2:
        return actions.copy()
    pad = window // 2
    padded = np.pad(actions, ((pad, pad), (0, 0)), mode='edge')
    smoothed = np.empty_like(actions)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    for dim in range(actions.shape[1]):
        smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode='valid')[: len(actions)]
    return smoothed


def _densify_action_keyframes(actions, max_translation_step, max_rotation_step):
    actions = np.asarray(actions, dtype=np.float64)
    if len(actions) <= 1:
        return actions.copy()
    dense = [actions[0].copy()]
    for start, end in zip(actions[:-1], actions[1:]):
        pos_dist = float(np.linalg.norm(end[:3] - start[:3]))
        rot_dist = float(np.linalg.norm(end[3:6] - start[3:6]))
        pos_steps = int(np.ceil(pos_dist / max(max_translation_step, 1e-6)))
        rot_steps = int(np.ceil(rot_dist / max(max_rotation_step, 1e-6)))
        steps = max(pos_steps, rot_steps, 1)
        for idx in range(1, steps + 1):
            alpha = float(idx) / float(steps)
            dense.append(start * (1.0 - alpha) + end * alpha)
    return np.asarray(dense, dtype=np.float64)


def prepare_startouch_smooth_chunk(
    full_actions,
    current_qpos,
    previous_goal,
    mode,
    stride,
    smooth_window,
    goal_alpha,
    max_translation_step,
    max_rotation_step,
):
    actions = np.asarray(full_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 7:
        raise ValueError(f'action chunk must have shape (N, >=7), got {actions.shape}')
    actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)
    actions[:, 6] = np.clip(actions[:, 6], 0.0, 1.0)
    actions = _moving_average_actions(actions, smooth_window)

    if previous_goal is not None and len(actions) > 0 and 0.0 < goal_alpha < 1.0:
        actions[-1] = previous_goal * (1.0 - goal_alpha) + actions[-1] * goal_alpha

    mode = str(mode)
    if mode == 'full':
        keyframes = actions
    elif mode == 'stride':
        stride = max(int(stride), 1)
        indices = list(range(0, len(actions), stride))
        if not indices or indices[-1] != len(actions) - 1:
            indices.append(len(actions) - 1)
        keyframes = actions[indices]
    elif mode == 'endpoint':
        keyframes = actions[[0, -1]] if len(actions) > 1 else actions
    else:
        raise ValueError(f'unsupported startouch chunk mode: {mode}')

    if len(keyframes) == 0:
        return keyframes, None

    current = np.asarray(current_qpos, dtype=np.float64).copy()
    current[6] = keyframes[0, 6]
    keyframes = np.vstack([current, keyframes])
    keyframes = _densify_action_keyframes(keyframes, max_translation_step, max_rotation_step)
    return keyframes, keyframes[-1].copy()


STARTOUCH_SDK_PATH = os.environ.get(
    'STARTOUCH_SDK_PATH',
    '/Users/alanliu/openpi_depth/startouch_sdk/interface_py',
)
if STARTOUCH_SDK_PATH and os.path.isdir(STARTOUCH_SDK_PATH):
    sys.path.insert(0, STARTOUCH_SDK_PATH)
log_probe('[PROBE] import startouchclass: begin')

from startouchclass import SingleArm
from tcp_compensation import (
    flange_position_to_tcp,
    parse_tool_offset_xyz,
    tcp_position_to_flange,
)
log_probe('[PROBE] import startouchclass: done')


CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 1280
CAMERA_FPS = 100


def parse_pose_rad(s):
    values = [float(x.strip()) for x in s.split(',')]
    if len(values) != 6:
        raise ValueError('pose must contain 6 comma-separated values: x,y,z,roll,pitch,yaw')
    return values[:3], values[3:6]


def init_yu12_camera(dev, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open /dev/video{dev}')
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YU12'))
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    fcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc = ''.join([chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4)])
    print(f'FOURCC from driver /dev/video{dev}: {fcc}')
    return cap


def grab_rgb_latest(cap, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, flush_n=4):
    for _ in range(flush_n):
        cap.grab()
    ok, raw = cap.read()
    if not ok:
        raise RuntimeError('Camera read failed')
    yuv = np.ascontiguousarray(raw).reshape(height * 3 // 2, width)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_frame_to_target(frame, target_size):
    return cv2.resize(frame, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)


def get_current_state(arm, tcp_offset, tcp_debug=False):
    pos_flange, quat_wxyz = arm.get_ee_pose_quat()
    pos_tcp = flange_position_to_tcp(pos_flange, quat_wxyz, tcp_offset)
    cur_pos, cur_euler = arm.get_ee_pose_euler()
    gripper_open = float(np.clip(arm.get_gripper_position(), 0.0, 1.0))

    if tcp_debug:
        print(
            '[TCP DEBUG][obs] flange -> TCP | '
            f'{np.asarray(pos_flange).round(4).tolist()} -> '
            f'{np.asarray(pos_tcp).round(4).tolist()} | '
            f'offset={tcp_offset.tolist()}'
        )

    qpos = np.array([*pos_tcp, *cur_euler, gripper_open], dtype=np.float32)
    return qpos, np.asarray(cur_pos, dtype=np.float64), np.asarray(cur_euler, dtype=np.float64)


def action_to_startouch_command(action, tcp_offset, gripper_close_gain=1.0, tcp_debug=False):
    action = np.asarray(action, dtype=np.float64)
    target_tcp = action[:3]
    target_euler_rad = action[3:6]
    gripper_open_raw = float(np.clip(action[6], 0.0, 1.0))
    gripper_open = float(np.clip(1.0 - float(gripper_close_gain) * (1.0 - gripper_open_raw), 0.0, 1.0))
    target_flange = tcp_position_to_flange(
        target_tcp.tolist(),
        target_euler_rad.tolist(),
        tcp_offset,
    )
    if tcp_debug:
        print(
            '[TCP DEBUG][cmd] TCP -> flange | '
            f'{np.asarray(target_tcp).round(4).tolist()} -> '
            f'{np.asarray(target_flange).round(4).tolist()}'
        )
    return np.asarray(target_flange, dtype=np.float64), target_euler_rad, gripper_open


def euler_to_quaternion_wxyz(euler):
    roll, pitch, yaw = np.asarray(euler, dtype=np.float64)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def actions_to_joint_waypoints(arm, actions, tcp_offset, gripper_close_gain=1.0, tcp_debug=False):
    q_seed = np.asarray(arm.get_joint_positions(), dtype=np.float64)
    if q_seed.shape != (6,):
        raise RuntimeError(f'Startouch current joints must have 6 values, got {q_seed.shape}')

    waypoints = []
    grippers = []
    command_records = []
    for action in np.asarray(actions, dtype=np.float64):
        target_flange, target_euler, gripper_open = action_to_startouch_command(
            action,
            tcp_offset,
            gripper_close_gain=gripper_close_gain,
            tcp_debug=tcp_debug,
        )
        quat_wxyz = euler_to_quaternion_wxyz(target_euler)
        joints, ok = arm.solve_ik(
            target_flange.tolist(),
            quat_wxyz.tolist(),
            q_seed=q_seed.tolist(),
        )
        if not ok:
            raise RuntimeError(
                'Startouch solve_ik failed for waypoint '
                f'pos={np.round(target_flange, 4).tolist()} '
                f'euler={np.round(target_euler, 4).tolist()}'
            )
        q_seed = np.asarray(joints, dtype=np.float64)
        if q_seed.shape != (6,):
            raise RuntimeError(f'Startouch solve_ik returned invalid joint shape: {q_seed.shape}')
        waypoints.append(q_seed)
        grippers.append(float(gripper_open))
        command_records.append((target_flange, target_euler, gripper_open))

    return np.asarray(waypoints, dtype=np.float64), np.asarray(grippers, dtype=np.float64), command_records


def move_waypoint_chunk_single(
    arm,
    actions,
    tcp_offset,
    control_hz,
    time_scale,
    speed_percent,
    duration_points=None,
    gripper_close_gain=1.0,
    tcp_debug=False,
):
    waypoints, grippers, command_records = actions_to_joint_waypoints(
        arm,
        actions,
        tcp_offset,
        gripper_close_gain=gripper_close_gain,
        tcp_debug=tcp_debug,
    )
    duration_count = len(waypoints) if duration_points is None else int(duration_points)
    time_sec = None if speed_percent is not None else duration_count / float(control_hz) * float(time_scale)
    if hasattr(arm, 'move_joint_waypoints_with_gripper'):
        duration = arm.move_joint_waypoints_with_gripper(
            waypoints,
            grippers,
            time_sec=time_sec,
            speed_percent=speed_percent,
        )
    else:
        duration = arm.set_joint_waypoints(
            waypoints,
            time_sec=time_sec,
            speed_percent=speed_percent,
        )
        arm.setGripperPosition(float(grippers[-1]))
    return duration, command_records


class RollingWaypointUpdater:
    def __init__(
        self,
        arm,
        tcp_offset,
        control_hz,
        time_scale,
        speed_percent,
        switch_delay_sec,
        max_retries,
        max_time_scale,
        gripper_close_gain,
        tcp_debug=False,
    ):
        self.arm = arm
        self.tcp_offset = tcp_offset
        self.control_hz = float(control_hz)
        self.time_scale = float(time_scale)
        self.speed_percent = speed_percent
        self.switch_delay_sec = float(switch_delay_sec)
        self.max_retries = int(max_retries)
        self.max_time_scale = float(max_time_scale)
        self.gripper_close_gain = float(gripper_close_gain)
        self.tcp_debug = tcp_debug
        self.queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._loop, name='rolling-waypoint-updater', daemon=True)

    def start(self):
        if not hasattr(self.arm, 'update_joint_waypoint_chunk_with_gripper'):
            raise RuntimeError('Startouch SDK does not provide update_joint_waypoint_chunk_with_gripper')
        if hasattr(self.arm, 'arm') and not hasattr(self.arm.arm, 'update_joint_waypoint_chunk_with_gripper'):
            raise RuntimeError('Startouch pybind does not provide update_joint_waypoint_chunk_with_gripper')
        self.thread.start()

    def submit(self, actions, duration_points):
        item = (np.asarray(actions, dtype=np.float64), int(duration_points), time.monotonic())
        while True:
            try:
                self.queue.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def raise_if_failed(self):
        if self.error is not None:
            raise self.error

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                actions, duration_points, submit_t = self.queue.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                waypoints, grippers, _records = actions_to_joint_waypoints(
                    self.arm,
                    actions,
                    self.tcp_offset,
                    gripper_close_gain=self.gripper_close_gain,
                    tcp_debug=self.tcp_debug,
                )
                if len(waypoints) == 0:
                    continue
                if self.speed_percent is None:
                    time_sec = duration_points / self.control_hz * self.time_scale
                else:
                    time_sec = None
                duration = None
                attempt = 0
                while True:
                    try:
                        duration = self.arm.update_joint_waypoint_chunk_with_gripper(
                            waypoints,
                            grippers,
                            time_sec=time_sec,
                            speed_percent=self.speed_percent,
                            switch_delay_sec=self.switch_delay_sec,
                        )
                        break
                    except RuntimeError as exc:
                        if self.speed_percent is not None or attempt >= self.max_retries:
                            raise
                        multiplier = _trajectory_limit_time_multiplier(str(exc))
                        if multiplier is None:
                            raise
                        base_time_sec = duration_points / self.control_hz * self.time_scale
                        next_time_sec = time_sec * multiplier * 1.15
                        max_time_sec = base_time_sec * self.max_time_scale
                        if next_time_sec > max_time_sec:
                            raise
                        attempt += 1
                        print(
                            '[RollingWaypoint] trajectory too fast; retrying slower: '
                            f'time_sec {time_sec:.3f} -> {next_time_sec:.3f} '
                            f'(attempt {attempt}/{self.max_retries})',
                            flush=True,
                        )
                        time_sec = next_time_sec
                print(
                    '[RollingWaypoint] updated: '
                    f'points={len(waypoints)} original_points={duration_points} '
                    f'duration={duration} time_sec={time_sec} latency={time.monotonic() - submit_t:.3f}s',
                    flush=True,
                )
            except BaseException as exc:  # noqa: BLE001
                self.error = exc
                self.stop_event.set()


def interp_and_move_single(arm, start_pos, start_euler, target_pos, target_euler, gripper_open, step_size, dt):
    ps = np.asarray(start_pos, dtype=np.float64)
    pe = np.asarray(target_pos, dtype=np.float64)
    es = np.asarray(start_euler, dtype=np.float64)
    ee = np.asarray(target_euler, dtype=np.float64)
    dist = np.linalg.norm(pe - ps)
    steps = max(int(np.ceil(dist / max(step_size, 1e-6))), 1)

    ts = np.linspace(0.0, 1.0, steps + 1)
    pos_arr = np.linspace(ps, pe, steps + 1)
    euler_arr = np.outer(1.0 - ts, es) + np.outer(ts, ee)

    for i in range(steps + 1):
        arm.set_end_effector_pose_euler_raw(
            pos=pos_arr[i].tolist(),
            euler=euler_arr[i].tolist(),
        )
        time.sleep(dt)
    arm.setGripperPosition(float(gripper_open))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='K001_single_arm')
    parser.add_argument('--can', type=str, default='can0', help='Startouch CAN interface')
    parser.add_argument('--camera_dev', type=int, default=0, help='Camera device index, e.g. /dev/video0')
    parser.add_argument('--ckpt_name', type=str, default=None, help='Checkpoint name; defaults to config eval_ckpt_name')
    parser.add_argument('--init_pose', type=str, default='0.3,0.0,0.16,0.0,0.0,0.0',
                        help='Initial pose: x,y,z,roll,pitch,yaw, xyz in meters, rpy in radians')
    parser.add_argument('--tcp_offset', type=str, default='0.0,0.0,0.0',
                        help='Tool offset from flange origin to TCP in flange frame, meters')
    parser.add_argument('--tcp_debug', action='store_true')
    parser.add_argument('--gripper_close_gain', type=float, default=1.0,
                        help='Amplify gripper closing amount: 1 keeps original, >1 closes tighter')
    parser.add_argument('--num_action_steps', type=int, default=20)
    parser.add_argument('--skip_action_steps', type=int, default=3)
    parser.add_argument('--controller_mode', choices=['raw', 'waypoint', 'rolling_waypoint', 'startouch_smooth'], default='raw',
                        help='raw: servo; waypoint: blocking planner; rolling_waypoint: non-blocking planner update; startouch_smooth: Startouch-tuned controller')
    parser.add_argument('--waypoint_control_hz', type=float, default=15.0,
                        help='Nominal action rate for waypoint chunk time_sec')
    parser.add_argument('--waypoint_time_scale', type=float, default=1.0,
                        help='Multiply waypoint chunk time_sec; larger values move slower')
    parser.add_argument('--waypoint_speed_percent', type=float, default=None,
                        help='Optional Startouch waypoint speed_percent; if set, overrides waypoint_control_hz timing')
    parser.add_argument('--waypoint_endpoint_only', action='store_true',
                        help='Waypoint mode executes only first and last action of each chunk')
    parser.add_argument('--rolling_switch_delay', type=float, default=0.05,
                        help='Switch delay seconds for rolling_waypoint updates')
    parser.add_argument('--rolling_max_retries', type=int, default=2,
                        help='Auto-slowdown retries for rolling_waypoint trajectory limit errors')
    parser.add_argument('--rolling_max_time_scale', type=float, default=4.0,
                        help='Maximum auto-slowdown multiplier relative to requested rolling waypoint time')
    parser.add_argument('--startouch_chunk_mode', choices=['endpoint', 'stride', 'full'], default='endpoint',
                        help='startouch_smooth keyframe selection from model chunk')
    parser.add_argument('--startouch_stride', type=int, default=5,
                        help='Keep every Nth model action when --startouch_chunk_mode=stride')
    parser.add_argument('--startouch_smooth_window', type=int, default=5,
                        help='Moving-average window for startouch_smooth model actions')
    parser.add_argument('--startouch_goal_alpha', type=float, default=0.65,
                        help='EMA weight for new chunk goal in startouch_smooth; lower is smoother')
    parser.add_argument('--startouch_max_translation_step', type=float, default=0.025,
                        help='Max TCP translation gap between Startouch keyframes, meters')
    parser.add_argument('--startouch_max_rotation_step', type=float, default=0.25,
                        help='Max RPY gap between Startouch keyframes, radians')
    parser.add_argument('--raw_blend_steps', type=int, default=5,
                        help='Blend first N raw-mode actions from previous command to new chunk')
    parser.add_argument('--raw_action_alpha', type=float, default=0.7,
                        help='Raw-mode low-pass alpha in action space; 1 disables filtering')
    parser.add_argument('--extra_time', type=int, default=1500)
    parser.add_argument('--dt', type=float, default=0.025, help='Sleep time for each interpolation step')
    parser.add_argument('--interp_step_size', type=float, default=0.0025, help='Max translation step in meters')
    parser.add_argument('--wait_for_enter', action='store_true', help='Wait for Enter before rollout')
    parser.add_argument('--no_display', action='store_true', help='Disable camera preview')
    parser.add_argument('--log_json', type=str, default=None, help='Optional path to save rollout debug JSON log')
    args = parser.parse_args()

    task = args.task
    cfg = SINGLE_ARM_TASK_CONFIG.copy()
    policy_config = SINGLE_ARM_POLICY_CONFIG.copy()
    train_cfg = SINGLE_ARM_TRAIN_CONFIG
    device = os.environ['DEVICE']
    num_queries = policy_config['num_queries']

    image_height = cfg.get('cam_height', 224)
    image_width = cfg.get('cam_width', 224)
    resize_target = (image_height, image_width)

    log_probe(f'Using camera names: {cfg["camera_names"]}')
    log_probe(f'State dim: {cfg["state_dim"]}, Action dim: {cfg["action_dim"]}, Num queries: {num_queries}')
    log_probe(f'Controller mode: {args.controller_mode}')
    waypoint_modes = ('waypoint', 'rolling_waypoint', 'startouch_smooth')
    rolling_modes = ('rolling_waypoint', 'startouch_smooth')
    if args.controller_mode in waypoint_modes and policy_config['temporal_agg']:
        raise ValueError(f'{args.controller_mode} controller_mode currently requires temporal_agg=False')
    if args.controller_mode in waypoint_modes and args.waypoint_control_hz <= 0:
        raise ValueError('--waypoint_control_hz must be positive')
    if args.controller_mode in waypoint_modes and args.waypoint_time_scale <= 0:
        raise ValueError('--waypoint_time_scale must be positive')
    if args.rolling_switch_delay < 0:
        raise ValueError('--rolling_switch_delay must be non-negative')
    if args.rolling_max_retries < 0:
        raise ValueError('--rolling_max_retries must be non-negative')
    if args.rolling_max_time_scale < 1.0:
        raise ValueError('--rolling_max_time_scale must be >= 1')
    if args.gripper_close_gain <= 0:
        raise ValueError('--gripper_close_gain must be positive')
    if args.startouch_stride <= 0:
        raise ValueError('--startouch_stride must be positive')
    if args.startouch_smooth_window <= 0:
        raise ValueError('--startouch_smooth_window must be positive')
    if not (0.0 < args.startouch_goal_alpha <= 1.0):
        raise ValueError('--startouch_goal_alpha must be in (0, 1]')
    if args.startouch_max_translation_step <= 0:
        raise ValueError('--startouch_max_translation_step must be positive')
    if args.startouch_max_rotation_step <= 0:
        raise ValueError('--startouch_max_rotation_step must be positive')
    if args.raw_blend_steps < 0:
        raise ValueError('--raw_blend_steps must be non-negative')
    if not (0.0 < args.raw_action_alpha <= 1.0):
        raise ValueError('--raw_action_alpha must be in (0, 1]')

    tcp_offset = parse_tool_offset_xyz(args.tcp_offset)
    if np.any(tcp_offset != 0.0):
            log_probe(f'[INFO] TCP offset: {tcp_offset.tolist()}')

    arm = None
    cam = None
    prev_action = None
    last_commanded_action = None
    filtered_action = None
    action_chunk = None
    rolling_updater = None
    startouch_previous_goal = None
    rollout_log = None
    if args.log_json:
        rollout_log = {
            'task': task,
            'can': args.can,
            'camera_dev': args.camera_dev,
            'ckpt_name': args.ckpt_name or train_cfg['eval_ckpt_name'],
            'num_queries': int(num_queries),
            'num_action_steps': int(args.num_action_steps),
            'skip_action_steps': int(args.skip_action_steps),
            'controller_mode': args.controller_mode,
            'waypoint_control_hz': float(args.waypoint_control_hz),
            'waypoint_time_scale': float(args.waypoint_time_scale),
            'waypoint_speed_percent': None if args.waypoint_speed_percent is None else float(args.waypoint_speed_percent),
            'waypoint_endpoint_only': bool(args.waypoint_endpoint_only),
            'rolling_switch_delay': float(args.rolling_switch_delay),
            'rolling_max_retries': int(args.rolling_max_retries),
            'rolling_max_time_scale': float(args.rolling_max_time_scale),
            'startouch_chunk_mode': args.startouch_chunk_mode,
            'startouch_stride': int(args.startouch_stride),
            'startouch_smooth_window': int(args.startouch_smooth_window),
            'startouch_goal_alpha': float(args.startouch_goal_alpha),
            'startouch_max_translation_step': float(args.startouch_max_translation_step),
            'startouch_max_rotation_step': float(args.startouch_max_rotation_step),
            'raw_blend_steps': int(args.raw_blend_steps),
            'raw_action_alpha': float(args.raw_action_alpha),
            'extra_time': int(args.extra_time),
            'dt': float(args.dt),
            'interp_step_size': float(args.interp_step_size),
            'temporal_agg': bool(policy_config['temporal_agg']),
            'init_pose': [float(v) for v in args.init_pose.split(',')],
            'tcp_offset': _to_float_list(tcp_offset),
            'gripper_close_gain': float(args.gripper_close_gain),
            'steps': [],
        }
    try:
        init_pos, init_euler = parse_pose_rad(args.init_pose)

        log_probe(f'[INFO] Connecting Startouch arm on {args.can}')
        log_probe('[PROBE] SingleArm ctor: begin')
        arm = SingleArm(can_interface_=args.can)
        log_probe('[PROBE] SingleArm ctor: done')
        log_probe('[PROBE] sleep after ctor: begin')
        time.sleep(2)
        log_probe('[PROBE] sleep after ctor: done')

        log_probe(f'[INFO] Moving to init pose: {args.init_pose}')
        log_probe('[PROBE] move init pose: begin')
        arm.set_end_effector_pose_euler(pos=init_pos, euler=init_euler, tf=1)
        log_probe('[PROBE] move init pose: done')
        log_probe('[PROBE] open gripper: begin')
        arm.setGripperPosition(1.0)
        log_probe('[PROBE] open gripper: done')
        log_probe('[INFO] Init pose reached.')
        if args.controller_mode in rolling_modes:
            rolling_updater = RollingWaypointUpdater(
                arm,
                tcp_offset,
                control_hz=args.waypoint_control_hz,
                time_scale=args.waypoint_time_scale,
                speed_percent=args.waypoint_speed_percent,
                switch_delay_sec=args.rolling_switch_delay,
                max_retries=args.rolling_max_retries,
                max_time_scale=args.rolling_max_time_scale,
                gripper_close_gain=args.gripper_close_gain,
                tcp_debug=args.tcp_debug,
            )
            rolling_updater.start()
            log_probe(f'[INFO] Rolling waypoint updater started for {args.controller_mode}.')

        ckpt_name = args.ckpt_name or train_cfg['eval_ckpt_name']
        ckpt_path = os.path.join(train_cfg['checkpoint_dir'], task, ckpt_name)
        log_probe(f'Loaded: {ckpt_path}')
        log_probe('[PROBE] make_policy: begin')
        policy = make_policy(policy_config['policy_class'], policy_config)
        log_probe('[PROBE] make_policy: done')
        log_probe('[PROBE] torch.load ckpt: begin')
        loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
        log_probe('[PROBE] torch.load ckpt: done')
        log_probe(str(loading_status))
        log_probe('[PROBE] policy.to/eval: begin')
        policy.to(device)
        policy.eval()
        log_probe('[PROBE] policy.to/eval: done')

        stats_path = os.path.join(train_cfg['checkpoint_dir'], task, 'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        log_probe('[PROBE] init camera: begin')
        cam = init_yu12_camera(args.camera_dev)
        log_probe('[PROBE] init camera: done')
        for _ in range(60):
            _ = cam.read()
        log_probe('[INFO] Camera warm-up done.')

        if args.wait_for_enter:
            input('Press Enter to start rollout...')
        else:
            print('[INFO] Waiting 3 seconds before rollout...')
            time.sleep(3)

        image = grab_rgb_latest(cam)
        image = resize_frame_to_target(image, resize_target)
        cv2.imwrite('cam_startouch_single.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        print('#' * 50)
        print('READY TO START STARTOUCH SINGLE ARM!')
        print('#' * 50)

        for _ in range(1):
            if policy_config['temporal_agg']:
                all_time_actions = torch.zeros(
                    [cfg['episode_len'] + args.extra_time, cfg['episode_len'] + num_queries + args.extra_time, cfg['action_dim']]
                ).to(device)

            with torch.inference_mode():
                for t in range(cfg['episode_len'] + args.extra_time):
                    qpos, cur_flange_pos, cur_euler = get_current_state(arm, tcp_offset, args.tcp_debug)
                    print(
                        '【position】 Startouch: '
                        f'[{qpos[0]:.4f}, {qpos[1]:.4f}, {qpos[2]:.4f}, '
                        f'{qpos[3]:.4f}, {qpos[4]:.4f}, {qpos[5]:.4f}, {qpos[6]:.4f}]'
                    )

                    qpos_norm = pre_process(qpos)
                    qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(device)

                    image = grab_rgb_latest(cam)
                    image = resize_frame_to_target(image, resize_target)
                    if not args.no_display:
                        cv2.imshow('Startouch Single Camera', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord('q'), ord('Q')):
                            raise KeyboardInterrupt('User quit from camera window')

                    imgs_np = np.expand_dims(image, axis=0)
                    imgs_np = np.transpose(imgs_np, (0, 3, 1, 2))
                    imgs_np = imgs_np.astype(np.float32) / 255.0
                    curr_image = torch.from_numpy(imgs_np).unsqueeze(0).to(device)

                    if policy_config['temporal_agg']:
                        print('inference', t)
                        all_actions = policy(qpos_tensor, curr_image)
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.26
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        if t % args.num_action_steps == 0:
                            all_actions = policy(qpos_tensor, curr_image)
                            if args.controller_mode == 'raw':
                                action_chunk = post_process(all_actions[0].cpu().numpy())
                                if last_commanded_action is not None and args.raw_blend_steps > 0:
                                    blend_steps = min(
                                        int(args.raw_blend_steps),
                                        int(action_chunk.shape[0]) - 1,
                                        int(args.num_action_steps) - 1,
                                    )
                                    if blend_steps > 0:
                                        blend_target = action_chunk[blend_steps].copy()
                                        for i in range(blend_steps):
                                            alpha = float(i + 1) / float(blend_steps)
                                            action_chunk[i] = (
                                                last_commanded_action * (1.0 - alpha)
                                                + blend_target * alpha
                                            )
                        if args.controller_mode in waypoint_modes:
                            if t % args.num_action_steps != 0:
                                continue
                            chunk_start = min(args.skip_action_steps + 1, all_actions.shape[1])
                            raw_chunk = all_actions[0, chunk_start:args.num_action_steps].cpu().numpy()
                            full_waypoint_action_chunk = post_process(raw_chunk)
                            if len(full_waypoint_action_chunk) > 0:
                                if args.controller_mode == 'startouch_smooth':
                                    waypoint_action_chunk, startouch_previous_goal = prepare_startouch_smooth_chunk(
                                        full_waypoint_action_chunk,
                                        qpos,
                                        startouch_previous_goal,
                                        mode=args.startouch_chunk_mode,
                                        stride=args.startouch_stride,
                                        smooth_window=args.startouch_smooth_window,
                                        goal_alpha=args.startouch_goal_alpha,
                                        max_translation_step=args.startouch_max_translation_step,
                                        max_rotation_step=args.startouch_max_rotation_step,
                                    )
                                elif args.waypoint_endpoint_only and len(full_waypoint_action_chunk) > 1:
                                    waypoint_action_chunk = np.stack(
                                        [full_waypoint_action_chunk[0], full_waypoint_action_chunk[-1]],
                                        axis=0,
                                    )
                                else:
                                    waypoint_action_chunk = full_waypoint_action_chunk
                                duration_points = len(full_waypoint_action_chunk)
                                if args.controller_mode == 'startouch_smooth':
                                    duration_points = max(duration_points, len(waypoint_action_chunk))
                                if args.controller_mode in rolling_modes:
                                    rolling_updater.raise_if_failed()
                                    rolling_updater.submit(
                                        waypoint_action_chunk,
                                        duration_points=duration_points,
                                    )
                                    if rollout_log is not None:
                                        rollout_log['steps'].append({
                                            't': int(t),
                                            'qpos': _to_float_list(qpos),
                                            'qpos_norm': _to_float_list(qpos_norm),
                                            'raw_action_chunk': [_to_float_list(a) for a in raw_chunk],
                                            'full_action_chunk': [_to_float_list(a) for a in full_waypoint_action_chunk],
                                            'action_chunk': [_to_float_list(a) for a in waypoint_action_chunk],
                                            'waypoint_endpoint_only': bool(args.waypoint_endpoint_only),
                                            'startouch_chunk_mode': args.startouch_chunk_mode if args.controller_mode == 'startouch_smooth' else None,
                                            'should_execute': True,
                                            'rolling_submitted': True,
                                        })
                                    continue
                                duration, command_records = move_waypoint_chunk_single(
                                    arm,
                                    waypoint_action_chunk,
                                    tcp_offset,
                                    control_hz=args.waypoint_control_hz,
                                    time_scale=args.waypoint_time_scale,
                                    speed_percent=args.waypoint_speed_percent,
                                    duration_points=len(full_waypoint_action_chunk),
                                    gripper_close_gain=args.gripper_close_gain,
                                    tcp_debug=args.tcp_debug,
                                )
                                print(
                                    '[Waypoint] Startouch single chunk executed: '
                                    f'points={len(waypoint_action_chunk)} '
                                    f'original_points={len(full_waypoint_action_chunk)} '
                                    f'endpoint_only={args.waypoint_endpoint_only} '
                                    f'duration={duration}'
                                )
                                if rollout_log is not None:
                                    rollout_log['steps'].append({
                                        't': int(t),
                                        'qpos': _to_float_list(qpos),
                                        'qpos_norm': _to_float_list(qpos_norm),
                                        'raw_action_chunk': [_to_float_list(a) for a in raw_chunk],
                                        'full_action_chunk': [_to_float_list(a) for a in full_waypoint_action_chunk],
                                        'action_chunk': [_to_float_list(a) for a in waypoint_action_chunk],
                                        'waypoint_endpoint_only': bool(args.waypoint_endpoint_only),
                                        'target_flange_chunk': [_to_float_list(r[0]) for r in command_records],
                                        'target_euler_chunk': [_to_float_list(r[1]) for r in command_records],
                                        'gripper_chunk': [float(r[2]) for r in command_records],
                                        'waypoint_duration': float(duration),
                                        'should_execute': True,
                                    })
                            continue
                        raw_action = all_actions[:, t % args.num_action_steps]

                    raw_action = raw_action.squeeze(0).cpu().numpy()
                    if policy_config['temporal_agg']:
                        action = post_process(raw_action)
                    elif args.controller_mode == 'raw':
                        if action_chunk is None:
                            raise RuntimeError('raw action chunk is not initialized')
                        action = action_chunk[t % args.num_action_steps].copy()
                    else:
                        action = post_process(raw_action)
                    if args.controller_mode == 'raw' and args.raw_action_alpha < 1.0:
                        if filtered_action is None:
                            filtered_action = action.copy()
                        else:
                            filtered_action = (
                                filtered_action * (1.0 - args.raw_action_alpha)
                                + action * args.raw_action_alpha
                            )
                        action = filtered_action.copy()
                    target_flange, target_euler, gripper_open = action_to_startouch_command(
                        action,
                        tcp_offset,
                        gripper_close_gain=args.gripper_close_gain,
                        tcp_debug=args.tcp_debug,
                    )
                    delta_action_qpos = action - qpos
                    delta_action_prev = None if prev_action is None else (action - prev_action)

                    print(
                        f'【action step {t}】 Startouch TCP: '
                        f'[{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}, '
                        f'{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}], '
                        f'gripper: {gripper_open:.4f}'
                    )
                    print('#' * 50)

                    should_execute = t % args.num_action_steps > args.skip_action_steps
                    if rollout_log is not None:
                        rollout_log['steps'].append({
                            't': int(t),
                            'qpos': _to_float_list(qpos),
                            'qpos_norm': _to_float_list(qpos_norm),
                            'raw_action': _to_float_list(raw_action),
                            'action': _to_float_list(action),
                            'delta_action_qpos': _to_float_list(delta_action_qpos),
                            'delta_action_prev': None if delta_action_prev is None else _to_float_list(delta_action_prev),
                            'cur_flange_pos': _to_float_list(cur_flange_pos),
                            'cur_euler': _to_float_list(cur_euler),
                            'target_flange': _to_float_list(target_flange),
                            'target_euler': _to_float_list(target_euler),
                            'gripper_open': float(gripper_open),
                            'should_execute': bool(should_execute),
                        })
                    prev_action = np.asarray(action, dtype=np.float64)

                    if should_execute:
                        interp_and_move_single(
                            arm,
                            cur_flange_pos,
                            cur_euler,
                            target_flange,
                            target_euler,
                            gripper_open,
                            step_size=args.interp_step_size,
                            dt=args.dt,
                        )
                        last_commanded_action = np.asarray(action, dtype=np.float64).copy()

    finally:
        if rolling_updater is not None:
            rolling_updater.stop()
        if rollout_log is not None and args.log_json:
            log_path = os.path.abspath(os.path.expanduser(args.log_json))
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(rollout_log, f, ensure_ascii=False, indent=2)
            print(f'[INFO] Saved rollout JSON log to {log_path}')
        if cam is not None:
            cam.release()
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if arm is not None:
            arm.cleanup()
        print('[INFO] STOP.')


if __name__ == '__main__':
    main()
