#!/usr/bin/env python3
"""
单任务各 Episode 的 Action 动作曲线

功能:
  - 读取指定任务目录下所有 HDF5 文件的 action 数据
  - 每个 action 维度一个子图, 所有 episode 的轨迹叠加显示
  - 标注 ±π 参考线, 自动高亮越界维度
  - 支持按 episode 数量限制 (避免太多轨迹看不清)

用法:
    # 单任务
    python plot_action_curves.py --task_dir /path/to/data/turn_switch

    # 限制显示的 episode 数量
    python plot_action_curves.py --task_dir /path/to/data/turn_switch --max_episodes 20

    # 批量
    python plot_action_curves.py --root_dir /path/to/data
"""

import os
import argparse
import numpy as np
import h5py
from glob import glob


def collect_task_episodes(task_dir, data_mode="both"):
    """收集单个任务目录下所有 episode 的 action 数据, 按 episode 分开"""
    hdf5_files = []
    if data_mode in ["clean", "both"]:
        hdf5_files.extend(sorted(glob(os.path.join(task_dir, "demo_clean", "data", "*.hdf5"))))
    if data_mode in ["randomized", "both"]:
        hdf5_files.extend(sorted(glob(os.path.join(task_dir, "demo_randomized", "data", "*.hdf5"))))

    if not hdf5_files:
        print(f"  警告: {task_dir} 下未找到 HDF5 文件")
        return []

    episodes = []
    for fp in hdf5_files:
        try:
            with h5py.File(fp, 'r') as f:
                if 'joint_action/vector' in f:
                    action = f['joint_action']['vector'][:]  # (T, D)
                    name = os.path.splitext(os.path.basename(fp))[0]
                    episodes.append({"name": name, "action": action})
        except Exception as e:
            print(f"  跳过 {fp}: {e}")

    return episodes


def plot_task_curves(task_name, episodes, save_dir, max_episodes=50):
    """绘制单任务的动作曲线"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    if not episodes:
        return

    D = episodes[0]["action"].shape[1]
    PI = np.pi

    # 限制 episode 数量
    if len(episodes) > max_episodes:
        step = len(episodes) // max_episodes
        episodes_show = episodes[::step][:max_episodes]
        print(f"  显示 {len(episodes_show)}/{len(episodes)} episodes (每隔 {step} 个采样)")
    else:
        episodes_show = episodes

    n_eps = len(episodes_show)

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

    # 颜色映射
    cmap = get_cmap('tab20' if n_eps <= 20 else 'hsv')
    colors = [cmap(i / max(n_eps - 1, 1)) for i in range(n_eps)]

    # 布局: 每个维度一个子图, 7行2列
    n_cols = 2
    n_rows = (D + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.0 * n_rows))
    axes = axes.flatten()

    fig.suptitle(
        f"Action Trajectories: {task_name}\n"
        f"({n_eps} episodes shown, red dashed = ±π)",
        fontsize=14, fontweight='bold'
    )

    for d in range(D):
        ax = axes[d]
        is_gripper = (d in [6, 13])
        has_exceed = False

        for ei, ep in enumerate(episodes_show):
            traj = ep["action"][:, d]
            t = np.arange(len(traj))

            # 检测越界
            if not is_gripper and (traj.min() < -PI or traj.max() > PI):
                has_exceed = True

            ax.plot(t, traj, color=colors[ei], alpha=0.5, linewidth=0.6)

        # ±π 参考线
        if not is_gripper:
            ax.axhline(y=PI, color='red', linestyle='--', alpha=0.5, linewidth=1.0)
            ax.axhline(y=-PI, color='red', linestyle='--', alpha=0.5, linewidth=1.0)

        # 零线
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.2, linewidth=0.5)

        title = dim_labels[d]
        if has_exceed:
            title += "  ⚠ EXCEEDS ±π"
        ax.set_title(title, fontsize=10,
                     fontweight='bold' if has_exceed else 'normal',
                     color='red' if has_exceed else 'black')
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.2)

    for d in range(D, len(axes)):
        axes[d].set_visible(False)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"action_curves_{task_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Action 动作曲线可视化")
    parser.add_argument("--task_dir", type=str,
                        default='/home/lab347-no10/Tianyi/RoboTwin/data/beat_block_hammer',
                        help="单个任务目录路径")
    parser.add_argument("--root_dir", type=str, default=None,
                        help="数据根目录 (批量模式)")
    parser.add_argument("--data_mode", type=str, default="both",
                        choices=["clean", "randomized", "both"])
    parser.add_argument("--save_dir", type=str, default="./action_curve_plots")
    parser.add_argument("--max_episodes", type=int, default=100,
                        help="每个任务最多显示的 episode 数")
    args = parser.parse_args()

    if args.task_dir is None and args.root_dir is None:
        parser.error("必须指定 --task_dir 或 --root_dir")

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
        print(f"  任务: {task_name}")
        print(f"{'='*50}")

        episodes = collect_task_episodes(task_dir, args.data_mode)
        if not episodes:
            continue

        print(f"  共 {len(episodes)} 个 episodes, "
              f"长度范围: {min(e['action'].shape[0] for e in episodes)}"
              f"-{max(e['action'].shape[0] for e in episodes)} steps")

        plot_task_curves(task_name, episodes, args.save_dir, args.max_episodes)


if __name__ == "__main__":
    main()