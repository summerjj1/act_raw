# Sony ACT 评估文档索引

本目录是 `act_raw_2` 与 `act_raw_5` 的统一评估入口。原始 CSV、JSON 和自动生成的 Markdown 仍保存在各自的 `run/` 目录中。

## 当前主实验

| 项目 | raw_2 | raw_5 |
|---|---|---|
| 训练集 | episode 001–130 | episode 001–130 |
| 开环测试集 | episode 131–150（20 集） | episode 131–150（20 集） |
| 训练步数 | 40000 | 40000 |
| Checkpoint | `policy_last.ckpt` | `policy_last.ckpt` |
| 输入 | Sony 四通道线性 RAW | Sony 四通道线性 RAW |
| 通道顺序 | `sp1_high,sp1_low,sp2_high,sp2_low` | `sp1_high,sp1_low,sp2_high,sp2_low` |
| 图像归一化 | black=0, white=4095, gamma=1.0 | black=0, white=4095, gamma=1.0 |
| 状态/动作 | 8D quaternion | 8D quaternion |
| 查询长度 | 80 | 80 |

最新 raw_2/raw_5 对比和结论见 [SONY_RAW2_VS_RAW5_STEP40000.md](SONY_RAW2_VS_RAW5_STEP40000.md)。

历史实验及对应原始文件见 [HISTORY.md](HISTORY.md)。

## 原始结果位置

raw_2：

```text
/share/project/dy/dy1/code/act_raw/act_raw_2/run/
  pick_place_sony_raw2_sp1_sp2_linear_quat_ep001_130_step40000/
    policy_last/open_loop_eval_131_150/
      open_loop_eval.md
      open_loop_eval.csv
      open_loop_eval.summary.json
```

raw_5：

```text
/share/project/dy/dy1/code/act_raw/act_raw_5/run/
  pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_130_step40000/
    policy_last/open_loop_eval_131_150/
      open_loop_eval.md
      open_loop_eval.csv
      open_loop_eval.summary.json
```

## 文件用途

- `open_loop_eval.md`：适合快速查看聚合指标。
- `open_loop_eval.csv`：逐 episode 指标，用于定位异常 episode 和画图。
- `open_loop_eval.summary.json`：完整聚合指标、数据路径、checkpoint、归一化配置和评估元数据。

## 指标解释

- `First action`：在每个时间点只比较动作块中的第一个预测动作，更接近滚动执行时实际采用的动作。
- `Full chunk`：比较模型预测的完整动作块。预测时间越远误差通常越大，因此该指标一般高于 First action。
- `XYZ MAE`：位置三个分量的平均绝对误差；单位继承原机械臂数据。
- `Rotation angle MAE`：四元数对应的旋转角误差，单位为度。计算时已处理 `q` 与 `-q` 表示同一旋转的问题。
- `Gripper MAE`：夹爪数值平均绝对误差；单位继承原始 `gripper_position`。

这些结果是离线开环预测误差，不能直接等同于机器人闭环抓取成功率。不同测试 episode、训练集规模或训练步数的结果不应直接横向比较。
