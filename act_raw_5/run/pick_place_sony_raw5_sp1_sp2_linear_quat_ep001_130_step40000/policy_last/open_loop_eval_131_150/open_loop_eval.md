# Sony 开环测试结果

- Checkpoint: `/share/project/dy/dy1/code/act_raw/act_raw_5/checkpoints/pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_130_step40000/policy_last.ckpt`
- 测试集: episode 131–150（20 集）
- 图像编码: `sony_bin_uint16`，黑白电平 `0/4095`，gamma `1.0`
- 查询长度: 80

## 聚合指标

| 模式 | XYZ MAE | 旋转角 MAE (deg) | 四元数分量 MAE | Gripper MAE |
|---|---:|---:|---:|---:|
| First action | 0.004891 | 3.784 | 0.014254 | 4.157557 |
| Full chunk | 0.016908 | 6.963 | 0.026187 | 18.432379 |

逐 episode 指标见同目录 CSV；完整统计见 JSON。
