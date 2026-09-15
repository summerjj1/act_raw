# Sony 开环测试结果

- Checkpoint: `/share/project/dy/dy1/code/act_raw_2/checkpoints/pick_place_sony_raw2_sp1_sp2_linear_quat_ep001_050_step30000/policy_step_30000_seed_42.ckpt`
- 测试集: episode 51–60（10 集）
- 图像编码: `sony_bin_uint16`，黑白电平 `0/4095`，gamma `1.0`
- 查询长度: 80

## 聚合指标

| 模式 | XYZ MAE | 旋转角 MAE (deg) | 四元数分量 MAE | Gripper MAE |
|---|---:|---:|---:|---:|
| First action | 0.004356 | 3.853 | 0.014490 | 4.754664 |
| Full chunk | 0.018241 | 7.980 | 0.029812 | 22.867342 |

逐 episode 指标见同目录 CSV；完整统计见 JSON。
