#!/usr/bin/env python3
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW2 = Path("/share/project/dy/dy1/code/act_raw_2/run/pick_place_sony_raw2_sp1_sp2_linear_quat_ep001_050_step36000/policy_last/open_loop_eval_51_60/open_loop_eval.csv")
RAW5 = Path("/share/project/dy/dy1/code/act_raw_5/run/pick_place_sony_raw5_sp1_sp2_linear_quat_ep001_050_step36000/policy_last/open_loop_eval_51_60/open_loop_eval.csv")
OUT_DIR = RAW5.parent

def load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["episode_id"] = df["episode"].str.extract(r"(\d+)").astype(int)
    df["model"] = label
    return df

raw2 = load(RAW2, "RAW2")
raw5 = load(RAW5, "RAW5")
colors = {"RAW2": "#1f77b4", "RAW5": "#d62728"}
metrics = [
    ("xyz_mae", "XYZ MAE", "first_xyz_mae", "chunk_xyz_mae"),
    ("rotation_mae", "Rotation angle MAE (°)", "first_rotation_angle_mae_deg", "chunk_rotation_angle_mae_deg"),
    ("quaternion_mae", "Quaternion component MAE", "first_quaternion_component_mae", "chunk_quaternion_component_mae"),
    ("gripper_mae", "Gripper MAE", "first_gripper_mae", "chunk_gripper_mae"),
]

for mode, mode_title, columns, filename in (
    ("first", "First Action", [m[2] for m in metrics], "raw2_vs_raw5_first_action_mae_scatter.png"),
    ("chunk", "Full Chunk", [m[3] for m in metrics], "raw2_vs_raw5_full_chunk_mae_scatter.png"),
):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharex=True)
    for ax, (_, title, _, _), value_col in zip(axes, metrics, columns):
        for name, df in (("RAW2", raw2), ("RAW5", raw5)):
            ax.scatter(df["episode_id"], df[value_col], s=58, alpha=0.85,
                       color=colors[name], label=name, edgecolors="white", linewidths=0.5)
        ax.set_title(title)
        ax.set_xlabel("Test episode")
        ax.set_xticks(sorted(raw2["episode_id"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("MAE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, frameon=False)
    fig.suptitle(f"Sony RAW2 vs RAW5 — {mode_title} MAE Scatter Plot", fontsize=15, y=1.075)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = OUT_DIR / filename
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)
