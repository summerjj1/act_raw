# StarTouch Controller Notes

The model outputs a future action chunk in TCP pose space:

```text
[x, y, z, roll, pitch, yaw, gripper_open]
```

StarTouch execution is sensitive to how this chunk is interpreted. Direct raw servo is fast but exposes model jitter. Blocking waypoint execution is smooth but delays the next camera read and policy inference. The `startouch_smooth` mode is intended as the default online controller for StarTouch.

## startouch_smooth Pipeline

```text
policy action chunk
-> moving average
-> endpoint/stride/full keyframe selection
-> prepend current observed state
-> densify large TCP/RPY gaps
-> IK to joint waypoints
-> non-blocking update_joint_waypoint_chunk_with_gripper
-> auto-slowdown retry on SDK velocity-limit errors
```

## Tuning Order

1. If the robot shakes, increase smoothing:

```bash
--startouch_smooth_window 7
--startouch_goal_alpha 0.5
```

2. If the robot is too slow:

```bash
--waypoint_control_hz 20
--waypoint_time_scale 0.8
```

3. If endpoint-only loses task detail:

```bash
--startouch_chunk_mode stride --startouch_stride 5
```

4. If SDK reports joint velocity limits, increase:

```bash
--waypoint_time_scale 1.3
--rolling_max_time_scale 6
```
