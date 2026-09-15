# Sony Linear RAW2 与 RAW5 `policy_last.ckpt` 开环评估对比

## 1. 对比配置

| 项目 | RAW2 | RAW5 |
|---|---|---|
| 代码目录 | `act_raw_2` | `act_raw_5` |
| 权重 | `...raw2...step36000/policy_last.ckpt` | `...raw5...step36000/policy_last.ckpt` |
| 训练集 | episode 1–50（50 集） | episode 1–50（49 集，缺 episode 20） |
| 测试集 | episode 51–60（10 集） | episode 51–60（10 集） |
| 测试帧数 | 3000 | 3000 |
| 图像输入 | 4×224×224 | 4×224×224 |
| 图像编码/布局 | `sony_bin_uint16` / `CHW` | `sony_bin_uint16` / `CHW` |
| 通道顺序 | SP1/SP2 低、高增益四通道 | SP1/SP2 低、高增益四通道 |
| 预处理 | Linear，black=0，white=4095，gamma=1.0 | Linear，black=0，white=4095，gamma=1.0 |
| 图像 mean/std | `[0,0,0,0]` / `[1,1,1,1]` | `[0,0,0,0]` / `[1,1,1,1]` |
| 状态/动作 | 8D quaternion | 8D quaternion |
| 训练步数 | 36000 | 36000 |
| 查询长度 | 80 | 80 |

两套模型使用相同的网络结构和图像处理，但 RAW2 与 RAW5 来自不同采集批次，且 RAW5 的训练池缺少 episode 20。因此结果是“数据集 + 训练结果”的整体比较，不能单独归因于图像转换方式。

## 2. 图像处理一致性

训练端和评估端均执行：

```python
image = torch.clamp((image.float() - 0.0) / 4095.0, 0.0, 1.0)
# gamma == 1.0，不执行 pow
```

训练端单帧经 DataLoader 后为 `(B,1,4,H,W)`；评估端先读取 `(B,4,H,W)`，执行相同的缩放和裁剪，再 `unsqueeze(1)` 变为 `(B,1,4,H,W)`。策略内部的四通道 `Normalize(mean=0,std=1)` 为恒等变换。因此本次评估进入 backbone 的图像数值和维度与训练一致。

## 3. 聚合指标

“相对改善率”定义为 `(RAW2 - RAW5) / RAW2 × 100%`，正值表示 RAW5 误差更低。

| 模式 | 指标 | RAW2 | RAW5 | RAW5 - RAW2 | 相对改善率 | 更低者 |
|---|---|---:|---:|---:|---:|---|
| First | XYZ 平均 MAE | 0.004509 | 0.003425 | -0.001084 | +24.04% | RAW5 |
| First | 旋转角 MAE | 4.370° | 2.409° | -1.960° | +44.86% | RAW5 |
| First | 四元数分量平均 MAE | 0.016038 | 0.009145 | -0.006893 | +42.98% | RAW5 |
| First | Gripper MAE | 4.667104 | 2.720767 | -1.946337 | +41.70% | RAW5 |
| Full chunk | XYZ 平均 MAE | 0.018344 | 0.011203 | -0.007140 | +38.93% | RAW5 |
| Full chunk | 旋转角 MAE | 7.937° | 3.911° | -4.025° | +50.72% | RAW5 |
| Full chunk | 四元数分量平均 MAE | 0.029705 | 0.014852 | -0.014853 | +50.00% | RAW5 |
| Full chunk | Gripper MAE | 23.506027 | 15.320575 | -8.185452 | +34.82% | RAW5 |

## 4. 八维 First action MAE

| 维度 | RAW2 MAE | RAW5 MAE | RAW5 - RAW2 | 相对改善率 | 更低者 |
|---|---:|---:|---:|---:|---|
| x | 0.005628 | 0.005140 | -0.000488 | +8.68% | RAW5 |
| y | 0.003531 | 0.002252 | -0.001279 | +36.22% | RAW5 |
| z | 0.004368 | 0.002883 | -0.001485 | +33.99% | RAW5 |
| qx | 0.016785 | 0.008024 | -0.008761 | +52.20% | RAW5 |
| qy | 0.013384 | 0.009897 | -0.003487 | +26.06% | RAW5 |
| qz | 0.021388 | 0.010610 | -0.010778 | +50.39% | RAW5 |
| qw | 0.012594 | 0.008049 | -0.004545 | +36.09% | RAW5 |
| gripper | 4.667104 | 2.720767 | -1.946337 | +41.70% | RAW5 |

## 5. 八维 Full chunk MAE

| 维度 | RAW2 MAE | RAW5 MAE | RAW5 - RAW2 | 相对改善率 | 更低者 |
|---|---:|---:|---:|---:|---|
| x | 0.021018 | 0.015792 | -0.005226 | +24.86% | RAW5 |
| y | 0.013022 | 0.009003 | -0.004019 | +30.86% | RAW5 |
| z | 0.020990 | 0.008814 | -0.012177 | +58.01% | RAW5 |
| qx | 0.030709 | 0.013633 | -0.017076 | +55.61% | RAW5 |
| qy | 0.022508 | 0.015023 | -0.007486 | +33.26% | RAW5 |
| qz | 0.042301 | 0.017586 | -0.024715 | +58.43% | RAW5 |
| qw | 0.023303 | 0.013167 | -0.010136 | +43.50% | RAW5 |
| gripper | 23.506027 | 15.320575 | -8.185452 | +34.82% | RAW5 |

## 6. 结论与限制

- 在本次 episode 51–60 测试中，RAW5 在所有聚合指标和全部八个动作维度上都低于 RAW2。
- RAW5 的 First action 旋转角 MAE 降低约 44.86%，Full chunk 旋转角 MAE 降低约 50.72%，长序列预测稳定性更好。
- RAW5 的 Full chunk `z`、`qx`、`qz` 改善最明显；Gripper 误差也降低约 34.82%。
- RAW5 训练池缺少 episode 20，训练样本数比 RAW2 少 1 集；这次结果不能证明少一集数据带来收益。
- 两套测试集不是同一批原始采集内容，所以不能严格得出“RAW5 转换算法一定优于 RAW2”的结论。要隔离转换方式影响，应在同一场景/同一动作轨迹上交叉评估。
- `First action` 和 `Full chunk` 是离线动作误差指标，不等同于真实机器人闭环成功率。

## 7. 运行脚本和结果文件

```bash
bash /share/project/dy/dy1/code/act_raw_2/bash_tran_rawdata/open_eval_pick_place_sony_raw2_linear_quat_step36000_last.sh
bash /share/project/dy/dy1/code/act_raw_5/bash_tran_rawdata/open_eval_pick_place_sony_raw5_linear_quat_step36000_last.sh
```

评估输出位于各自项目：

```text
run/<task>/policy_last/open_loop_eval_51_60/
```

其中包括逐 episode 的 `open_loop_eval.csv`、聚合统计 `open_loop_eval.summary.json` 和简要 `open_loop_eval.md`。
