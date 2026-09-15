# Sony raw_2 与 raw_5 均匀抽样开环评估

## 实验设置

| 项目 | 设置 |
|---|---|
| 总 episode 数 | 150 |
| 外部测试集 | `10,20,30,40,50,60,70,80,90,100,110,120,130,140,150` |
| 训练池 | 除上述测试集外的 135 条 episode |
| 内部训练/验证 | 128/7 |
| 训练步数 | 40000 |
| 评估权重 | `policy_last.ckpt` |
| 查询长度 | 80 |
| 图像输入 | Sony uint16，4×224×224，线性 RAW |
| 通道顺序 | `sp1_high,sp1_low,sp2_high,sp2_low` |
| RAW 预处理 | `black=0, white=4095, gamma=1.0` |

## 聚合结果

`raw_5 相对变化`按 `(raw_2 - raw_5) / raw_2` 计算，正值表示 raw_5 误差更低。

| 预测模式 | 指标 | raw_2 | raw_5 | raw_5 相对变化 | 更低者 |
|---|---|---:|---:|---:|---|
| First action | XYZ MAE | 0.003081 | 0.003093 | -0.4% | raw_2，基本持平 |
| First action | 旋转角 MAE (deg) | 2.495 | 2.220 | +11.0% | raw_5 |
| First action | 四元数分量 MAE | 0.009291 | 0.008356 | +10.1% | raw_5 |
| First action | Gripper MAE | 3.4346 | 2.6894 | +21.7% | raw_5 |
| Full chunk | XYZ MAE | 0.014033 | 0.010929 | +22.1% | raw_5 |
| Full chunk | 旋转角 MAE (deg) | 5.566 | 4.755 | +14.6% | raw_5 |
| Full chunk | 四元数分量 MAE | 0.020875 | 0.017852 | +14.5% | raw_5 |
| Full chunk | Gripper MAE | 17.8232 | 12.4023 | +30.4% | raw_5 |

## 逐 episode 胜负

这里只统计同编号 episode 上哪个数值更低。raw_2 与 raw_5 是不同采集批次，因此同编号不代表同一物理轨迹。

| 指标 | raw_2 更低 | raw_5 更低 |
|---|---:|---:|
| First action XYZ | 8 | 7 |
| First action 旋转角 | 5 | 10 |
| First action 四元数分量 | 6 | 9 |
| First action Gripper | 4 | 11 |
| Full chunk XYZ | 6 | 9 |
| Full chunk 旋转角 | 6 | 9 |
| Full chunk 四元数分量 | 6 | 9 |
| Full chunk Gripper | 5 | 10 |

## 高误差 episode

### raw_2

raw_2 的 episode 20 在多项指标上明显最高：

| 指标 | episode 20 | 第二高值 |
|---|---:|---:|
| First action XYZ | 0.005842 | 0.004312（episode 110） |
| First action 旋转角 (deg) | 4.743 | 3.993（episode 110） |
| First action Gripper | 7.5444 | 4.4217（episode 30） |
| Full chunk XYZ | 0.034773 | 0.021111（episode 50） |
| Full chunk 旋转角 (deg) | 13.042 | 9.561（episode 50） |
| Full chunk Gripper | 43.1206 | 30.9273（episode 30） |

如果两组同时排除 episode 20，raw_2 的 First-action XYZ 为 0.002884，低于 raw_5 的 0.003115；raw_5 的其余七项指标仍然更低。该结果只作为敏感性检查，主结果仍应保留全部 15 条测试数据。

### raw_5

- episode 130：First-action XYZ、Gripper及 Full-chunk XYZ、旋转角、Gripper均靠前。
- episode 140：First-action 旋转角最高。
- episode 150：First-action XYZ 最高。

## 训练日志

| 数据 | 最小内部验证 loss | 所在 epoch | 最后一个 epoch 的训练 loss |
|---|---:|---:|---:|
| raw_2 | 0.128306 | 4100 | 0.109338 |
| raw_5 | 0.065328 | 4100 | 0.122225 |

`policy_best.ckpt` 与 `policy_last.ckpt` 不是同一份权重。本报告按实验约定评估 `policy_last.ckpt`，不能用最小验证 loss 直接推断 `policy_best.ckpt` 的开环结果。

## 结论

在这次 135/15 划分和 `policy_last.ckpt` 开环评估中，raw_5 在旋转、夹爪和 Full-chunk 长序列预测上整体更好，尤其 Full-chunk Gripper MAE 降低约 30.4%。First-action XYZ 几乎持平，raw_2 低约 0.4%。

由于 raw_2 与 raw_5 来自不同采集批次，相同 episode 编号并非同一物理动作，本结果代表“各自数据集训练并在各自测试集评估”的整体表现，不能单独证明某一路 RAW 图像方案更优。

## 结果位置

raw_2：

```text
/share/project/dy/dy1/code/act_raw/act_raw_2/run/pick_place_sony_raw2_sp1_sp2_linear_quat_train135_test15_every10_step40000/policy_last/open_loop_eval_test_every10_15eps/
```

raw_5：

```text
/share/project/dy/dy1/code/act_raw/act_raw_5/run/pick_place_sony_raw5_sp1_sp2_linear_quat_train135_test15_every10_step40000/policy_last/open_loop_eval_test_every10_15eps/
```
