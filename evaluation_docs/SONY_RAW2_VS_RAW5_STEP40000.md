# Sony raw_2 与 raw_5 开环评估对比

## 实验条件

两组实验均使用 episode 001–130 训练，episode 131–150 开环测试，训练 40000 步并评估 `policy_last.ckpt`。模型输入为四通道 Sony `uint16` 线性 RAW：

```text
sp1_high, sp1_low, sp2_high, sp2_low
black=0, white=4095, gamma=1.0
image_mean=[0,0,0,0], image_std=[1,1,1,1]
```

## 聚合结果

| 模式 | 数据 | XYZ MAE ↓ | 旋转角 MAE ↓ | 四元数分量 MAE ↓ | Gripper MAE ↓ |
|---|---|---:|---:|---:|---:|
| First action | raw_2 | **0.003175** | **3.448°** | **0.013062** | **3.0011** |
| First action | raw_5 | 0.004891 | 3.784° | 0.014254 | 4.1576 |
| Full chunk | raw_2 | **0.013458** | **6.313°** | **0.023282** | **16.8230** |
| Full chunk | raw_5 | 0.016908 | 6.963° | 0.026187 | 18.4324 |

在这组相同训练规模和评估协议下，raw_2 的所有聚合 MAE 均低于 raw_5：

- First action：XYZ 低约 35.1%，旋转角低约 8.9%，Gripper 低约 27.8%。
- Full chunk：XYZ 低约 20.4%，旋转角低约 9.3%，Gripper 低约 8.7%。

因此，当前离线开环指标支持 raw_2 优于 raw_5。该结论仅适用于当前数据划分、40000 步 checkpoint 和现有归一化配置。

## 高误差 episode

这里的“高误差”表示该 episode 的模型预测误差在 20 个测试 episode 中靠前，不代表原始数据已经确认异常或损坏。

| 数据 | 指标 | episode | 数值 |
|---|---|---|---:|
| raw_2 | First action XYZ MAE | 134 | 0.004855 |
| raw_2 | First action Rotation MAE | 134 | 6.571° |
| raw_2 | First action Gripper MAE | 134 | 7.831 |
| raw_2 | Full chunk XYZ MAE | 134 | 0.026686 |
| raw_2 | Full chunk Rotation MAE | 134 | 12.199° |
| raw_2 | Full chunk Gripper MAE | 134 | 43.046 |
| raw_5 | First action XYZ MAE | 141 | 0.007287 |
| raw_5 | First action Rotation MAE | 141 | 7.425° |
| raw_5 | First action Gripper MAE | 136 | 6.131 |
| raw_5 | Full chunk XYZ MAE | 141 | 0.038306 |
| raw_5 | Full chunk Rotation MAE | 141 | 17.622° |
| raw_5 | Full chunk Gripper MAE | 136 | 32.230 |

进一步统计没有发现图像全黑、零值或 4095 饱和。raw_2 episode 134 和 raw_5 episode 136 的夹爪变化次数分别为测试集最高；raw_5 episode 141 的位置轨迹范围为测试集最大。它们更可能是动作更复杂或模型泛化较弱的样本，仍建议结合录像或图像序列确认是否存在遮挡和时间对齐问题。

## 原始结果

raw_2：

```text
/share/project/dy/dy1/code/act_raw/act_raw_2/run/pick_place_sony_raw2_sp1_sp2_linear_quat_ep001_130_step40000/policy_last/open_loop_eval_131_150/
```

raw_5：

```text
/share/project/dy/dy1/code/act_raw/act_raw_5/run/pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_130_step40000/policy_last/open_loop_eval_131_150/
```
