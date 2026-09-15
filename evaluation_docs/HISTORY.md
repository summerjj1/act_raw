# Sony ACT 历史评估索引

## 当前可直接对比的实验

| 数据 | 训练 episode | 步数 | 测试 episode | First XYZ | First Rotation | First Gripper | Full XYZ | Full Rotation | Full Gripper |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| raw_2 linear | 001–130 | 40000 | 131–150 | 0.003175 | 3.448° | 3.001 | 0.013458 | 6.313° | 16.823 |
| raw_5 linear | 001–130 | 40000 | 131–150 | 0.004891 | 3.784° | 4.158 | 0.016908 | 6.963° | 18.432 |

## 早期实验

以下实验训练集只有 episode 001–050，测试集为 051–060，与当前 130/20 划分不同，只作为历史记录。

| 数据/配置 | 步数或 checkpoint | First XYZ | First Rotation | First Gripper | Full XYZ | Full Rotation | Full Gripper |
|---|---|---:|---:|---:|---:|---:|---:|
| raw_2 gamma 2.2 | step 30000 | 0.007416 | 5.113° | 6.259 | 0.021566 | 9.539° | 23.719 |
| raw_2 linear | step 30000 | 0.008033 | 4.899° | 5.643 | 0.025678 | 12.840° | 25.120 |
| raw_2 linear norm-fixed | step 30000 | 0.004356 | 3.853° | 4.755 | 0.018241 | 7.980° | 22.867 |
| raw_2 linear | policy_last, 36000 | 0.004509 | 4.370° | 4.667 | 0.018344 | 7.937° | 23.506 |
| raw_5 linear | step 30000 | 0.006957 | 4.979° | 4.866 | 0.019822 | 7.021° | 18.994 |
| raw_5 linear | policy_last, 36000 | 0.003425 | 2.409° | 2.721 | 0.011203 | 3.911° | 15.321 |

早期结果不能用于判断当前 raw_2 与 raw_5 的最终优劣，因为训练集规模、测试 episode、checkpoint 和部分归一化配置不同。

## 历史结果目录

```text
/share/project/dy/dy1/code/act_raw/act_raw_2/run/
/share/project/dy/dy1/code/act_raw/act_raw_5/run/
```

运行目录中的 CSV、JSON、Markdown 是原始记录。本索引仅汇总，不替代原始文件。
