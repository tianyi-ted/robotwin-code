#!/usr/bin/env python3
"""
单任务 Action 数值范围分布直方图

功能:
  - 读取指定任务目录下所有 HDF5 文件的 action 数据
  - 对每个 action 维度画直方图, 标注 min/max/mean 和 ±π 参考线
  - 自动检测角度越界 (超出 [-π, π]) 的维度并高亮

用法:
    # 单任务
    python plot_action_dist.py --task_dir /path/to/data/turn_switch

    # 指定数据模式
    python plot_action_dist.py --task_dir /path/to/data/turn_switch --data_mode both

    # 批量: 对 data 目录下所有任务
    python plot_action_dist.py --root_dir /path/to/data --data_mode both
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob

def collect_task_actions(task_dir, data_mode="both"):
    """收集单个任务目录下所有 episode 的 action 数据"""
    hdf5_files = []
    if data_mode in ["clean", "both"]:
        hdf5_files.extend(glob(os.path.join(task_dir, "demo_clean", "data", "*.hdf5")))
    if data_mode in ["randomized", "both"]:
        hdf5_files.extend(glob(os.path.join(task_dir, "demo_randomized", "data", "*.hdf5")))

    if not hdf5_files:
        print(f"  警告: {task_dir} 下未找到 HDF5 文件")
        return None

    all_actions = []
    for fp in hdf5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'joint_action/vector' in f:
                    all_actions.append(f['joint_action']['vector'][:])
        except Exception as e:
            print(f"  跳过 {fp}: {e}")

    if not all_actions:
        return None

    return np.concatenate(all_actions, axis=0)  # (total_timesteps, action_dim)


def plot_single_task(task_name, actions, save_dir, bins=200):
    """为单个任务绘制 action 分布直方图"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    T, D = actions.shape
    PI = np.pi

    # 维度标签
    dim_labels = []
    for i in range(D):
        if i == 6:
            dim_labels.append(f"D{i} (L gripper)")
        elif i == 13:
            dim_labels.append(f"D{i} (R gripper)")
        elif i < 7:
            dim_labels.append(f"D{i} (L joint {i})")
        else:
            dim_labels.append(f"D{i} (R joint {i-7})")

    # 布局: 7行2列 (适合14维)
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.2 * n_rows))
    axes = axes.flatten()

    fig.suptitle(
        f"Action Value Distribution: {task_name}\n"
        f"({T} timesteps, {D} dims, red dashed = ±π)",
        fontsize=14, fontweight='bold'
    )

    for d in range(D):
        ax = axes[d]
        data = actions[:, d]
        d_min, d_max = data.min(), data.max()
        d_mean = data.mean()
        d_std = data.std()

        # 检测是否超出 [-π, π]
        is_gripper = (d in [6, 13])
        exceeds_pi = (not is_gripper) and (d_min < -PI or d_max > PI)

        # 直方图颜色
        color = '#d62728' if exceeds_pi else '#1f77b4'
        edge_color = '#a02020' if exceeds_pi else '#1a5f90'

        # 过滤掉接近 0 的静止样本, 单独统计
        zero_thresh = 1e-4
        n_zero = np.sum(np.abs(data) < zero_thresh)
        data_nonzero = data[np.abs(data) >= zero_thresh]

        if len(data_nonzero) > 0:
            ax.hist(data_nonzero, bins=bins, color=color, alpha=0.7,
                    edgecolor=edge_color, linewidth=0.3)
        else:
            ax.hist(data, bins=bins, color=color, alpha=0.7,
                    edgecolor=edge_color, linewidth=0.3)

        # 对数 y 轴, 让低频区域也能看清
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.8)

        # ±π 参考线 (仅对关节维度)
        if not is_gripper:
            ax.axvline(x=PI, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
            ax.axvline(x=-PI, color='red', linestyle='--', alpha=0.6, linewidth=1.2)

        # 统计标注
        pct_zero = n_zero / len(data) * 100
        stats_text = f"min={d_min:.3f}  max={d_max:.3f}\nmean={d_mean:.3f}  std={d_std:.3f}"
        if pct_zero > 1:
            stats_text += f"\n({n_zero} zero samples filtered, {pct_zero:.0f}%)"
        if exceeds_pi:
            n_exceed = np.sum((data < -PI) | (data > PI))
            pct = n_exceed / len(data) * 100
            stats_text += f"\n⚠ {n_exceed} samples ({pct:.1f}%) outside ±π"

        ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8),
                family='monospace')

        title = dim_labels[d]
        if exceeds_pi:
            title += "  ⚠ EXCEEDS ±π"
        ax.set_title(title, fontsize=10, fontweight='bold' if exceeds_pi else 'normal',
                     color='red' if exceeds_pi else 'black')
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.2)

    # 隐藏多余的子图
    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"action_dist_{task_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")

    # 打印汇总
    print(f"\n  [{task_name}] {T} timesteps, {D} dims")
    for d in range(D):
        data = actions[:, d]
        is_gripper = (d in [6, 13])
        flag = ""
        if not is_gripper:
            n_exceed = np.sum((data < -PI) | (data > PI))
            if n_exceed > 0:
                flag = f"  ⚠ {n_exceed} ({n_exceed/len(data)*100:.1f}%) outside ±π"
        print(f"    {dim_labels[d]:20s}  min={data.min():+8.4f}  max={data.max():+8.4f}  "
              f"mean={data.mean():+8.4f}  std={data.std():.4f}{flag}")


def main():
    parser = argparse.ArgumentParser(description="Action 数值分布直方图")
    parser.add_argument("--task_dir", type=str, default='/home/lab347-no10/Tianyi/RoboTwin/data/beat_block_hammer',
                        help="单个任务目录路径 (如 /path/to/data/turn_switch)")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="数据根目录, 包含多个任务文件夹 (批量模式)")
    parser.add_argument("--data_mode", type=str, default="both",
                        choices=["clean", "randomized", "both"])
    parser.add_argument("--save_dir", type=str, default="./action_dist_plots")
    parser.add_argument("--bins", type=int, default=200)
    args = parser.parse_args()

    if args.task_dir is None and args.root_dir is None:
        parser.error("必须指定 --task_dir 或 --root_dir")

    # 收集要处理的任务列表
    tasks = []
    if args.task_dir:
        task_name = os.path.basename(args.task_dir.rstrip('/'))
        tasks.append((task_name, args.task_dir))
    elif args.root_dir:
        for d in sorted(os.listdir(args.root_dir)):
            full = os.path.join(args.root_dir, d)
            if os.path.isdir(full):
                has_data = (os.path.exists(os.path.join(full, "demo_clean")) or
                            os.path.exists(os.path.join(full, "demo_randomized")))
                if has_data:
                    tasks.append((d, full))
        print(f"找到 {len(tasks)} 个任务")

    for task_name, task_dir in tasks:
        print(f"\n{'='*50}")
        print(f"  处理任务: {task_name}")
        print(f"{'='*50}")

        actions = collect_task_actions(task_dir, args.data_mode)
        if actions is None:
            continue

        plot_single_task(task_name, actions, args.save_dir, bins=args.bins)


if __name__ == "__main__":
    main()