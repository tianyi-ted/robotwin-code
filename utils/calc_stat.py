#!/usr/bin/env python3
import os
import json
import numpy as np
import h5py
import multiprocessing as mp
from glob import glob
from tqdm import tqdm
import time
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_file(file_path):
    """
    处理单个 HDF5 文件，分别提取 Action 和 State 的统计信息
    (保持不变)
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 1. 提取 Action (T, 14)
            if 'joint_action/vector' not in f:
                return (file_path, False, None, None, None, None, "Missing joint_action/vector", [], [])
            
            action_data = f['joint_action']['vector'][:] 
            
            # 2. 提取 State (T, 16)
            try:
                l_pose = f['endpose']['left_endpose'][:]
                l_grip = f['endpose']['left_gripper'][:]
                r_pose = f['endpose']['right_endpose'][:]
                r_grip = f['endpose']['right_gripper'][:]
                
                if l_grip.ndim == 1: l_grip = l_grip.reshape(-1, 1)
                if r_grip.ndim == 1: r_grip = r_grip.reshape(-1, 1)
                
                state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            except KeyError as e:
                return (file_path, False, None, None, None, None, f"Missing state key: {e}", [], [])

        # --- 处理 Action 统计 ---
        act_min = np.min(action_data, axis=0)
        act_max = np.max(action_data, axis=0)
        
        act_outlier_dims = []
        if np.any(np.abs(action_data) > 3):
            act_outlier_dims = np.where(np.any(np.abs(action_data) > 3, axis=0))[0].tolist()

        # --- 处理 State 统计 ---
        state_min = np.min(state_data, axis=0)
        state_max = np.max(state_data, axis=0)
        
        state_outlier_dims = []
        if np.any(np.abs(state_data) > 5):
            state_outlier_dims = np.where(np.any(np.abs(state_data) > 5, axis=0))[0].tolist()

        return (file_path, True, act_min, act_max, state_min, state_max, None, act_outlier_dims, state_outlier_dims)
                
    except Exception as e:
        return (file_path, False, None, None, None, None, str(e), [], [])

def process_task_files(task_info):
    """
    处理单个任务目录下的所有文件
    
    Args:
        task_info: tuple of (task_name, task_dir, data_mode)
    """
    task_name, task_dir, data_mode = task_info
    
    hdf5_files = []
    
    # 根据 data_mode 构建搜索路径
    # 模式1: clean
    if data_mode in ["clean", "both"]:
        pattern = os.path.join(task_dir, "demo_clean", "data", "*.hdf5")
        hdf5_files.extend(glob(pattern))
        
    # 模式2: randomized
    if data_mode in ["randomized", "both"]:
        pattern = os.path.join(task_dir, "demo_randomized", "data", "*.hdf5")
        hdf5_files.extend(glob(pattern))

    results = {
        'task_name': task_name,
        'file_count': 0,
        'success_count': 0,
        'error_files': [],
        'outlier_files': [],
        'act_global_min': None,
        'act_global_max': None,
        'state_global_min': None,
        'state_global_max': None
    }

    for file_path in hdf5_files:
        (fpath, success, 
         act_min, act_max, state_min, state_max, 
         error_msg, act_outliers, state_outliers) = process_single_file(file_path)
        
        results['file_count'] += 1
        
        if success:
            results['success_count'] += 1
            
            # Update Action Stats
            if results['act_global_min'] is None:
                results['act_global_min'] = act_min
                results['act_global_max'] = act_max
            else:
                results['act_global_min'] = np.minimum(results['act_global_min'], act_min)
                results['act_global_max'] = np.maximum(results['act_global_max'], act_max)

            # Update State Stats
            if results['state_global_min'] is None:
                results['state_global_min'] = state_min
                results['state_global_max'] = state_max
            else:
                results['state_global_min'] = np.minimum(results['state_global_min'], state_min)
                results['state_global_max'] = np.maximum(results['state_global_max'], state_max)
            
            if act_outliers:
                results['outlier_files'].append((fpath, "Action", act_outliers))
            if state_outliers:
                results['outlier_files'].append((fpath, "State", state_outliers))
        else:
            results['error_files'].append((fpath, error_msg))
            
    return results

def collect_stats_multiprocess(root_dirs, output_path, outlier_path, num_processes=16, data_mode="clean"):
    """
    主函数：多进程统计 HDF5 数据集
    """
    # 最小修改：如果输入的是单个字符串路径，将其转为列表以统一处理
    if isinstance(root_dirs, str):
        root_dirs = [root_dirs]

    print(f"Starting stats calculation for HDF5 dataset in: {root_dirs}")
    print(f"Data Mode: {data_mode}")
    
    # 1. 扫描任务目录并准备参数
    task_dirs = []
    
    # 新增外层循环，遍历所有传入的 root_dir
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            # 将 Error 换成 Warning 并使用 continue，避免一个无效路径中断整个任务
            print(f"Warning: Root directory {root_dir} does not exist. Skipping.")
            continue

        raw_dirs = os.listdir(root_dir)
        for d in raw_dirs:
            full_path = os.path.join(root_dir, d)
            if os.path.isdir(full_path):
                # 简单的检查，看是否包含数据文件夹
                has_clean = os.path.exists(os.path.join(full_path, "demo_clean"))
                has_rand = os.path.exists(os.path.join(full_path, "demo_randomized"))
                
                if has_clean or has_rand:
                    # 将 data_mode 打包进参数 Tuple
                    task_dirs.append((d, full_path, data_mode))
    
    print(f"Found {len(task_dirs)} task directories.")
    
    # 2. 多进程处理
    start_time = time.time()
    with mp.Pool(processes=num_processes) as pool:
        task_results = list(tqdm(pool.imap(process_task_files, task_dirs), total=len(task_dirs), desc="Scanning Tasks"))

    # 3. 聚合结果
    total_files = 0
    total_success = 0
    all_errors = []
    all_outliers = []
    
    act_min = None; act_max = None
    state_min = None; state_max = None

    for res in task_results:
        total_files += res['file_count']
        total_success += res['success_count']
        all_errors.extend(res['error_files'])
        all_outliers.extend(res['outlier_files'])
        
        # Aggregate Action
        if res['act_global_min'] is not None:
            if act_min is None:
                act_min = res['act_global_min']
                act_max = res['act_global_max']
            else:
                act_min = np.minimum(act_min, res['act_global_min'])
                act_max = np.maximum(act_max, res['act_global_max'])

        # Aggregate State
        if res['state_global_min'] is not None:
            if state_min is None:
                state_min = res['state_global_min']
                state_max = res['state_global_max']
            else:
                state_min = np.minimum(state_min, res['state_global_min'])
                state_max = np.maximum(state_max, res['state_global_max'])

    elapsed_time = time.time() - start_time

    # 4. 构造输出字典
    stat_dict = {
        "robotwin2": {
            "meta": {
                "data_mode": data_mode,
                "total_files": total_files,
                "valid_files": total_success,
                "processing_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "action": {
                "min": act_min.tolist() if act_min is not None else [],
                "max": act_max.tolist() if act_max is not None else [],
                "dim": len(act_min) if act_min is not None else 0
            },
            "state": {
                "min": state_min.tolist() if state_min is not None else [],
                "max": state_max.tolist() if state_max is not None else [],
                "dim": len(state_min) if state_min is not None else 0
            }
        }
    }

    # 5. 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stat_dict, f, indent=4)
        print(f"\nStats saved to: {output_path}")
    except Exception as e:
        print(f"Error saving stats json: {e}")

    try:
        with open(outlier_path, 'w', encoding='utf-8') as f:
            f.write(f"Outlier Report - Mode: {data_mode} - Total: {len(all_outliers)}\n")
            f.write("="*60 + "\n")
            f.write("Thresholds: Action > 3.0, State > 5.0\n\n")
            for item in all_outliers:
                f.write(f"[{item[1]}] {item[0]} -> Dims: {item[2]}\n")
        print(f"Outlier report saved to: {outlier_path}")
    except Exception as e:
        print(f"Error saving outlier report: {e}")

    # 6. 打印摘要
    print(f"\n{'='*30} SUMMARY {'='*30}")
    print(f"Mode: {data_mode}")
    print(f"Processed {total_success}/{total_files} files in {elapsed_time:.2f}s")
    if act_min is not None:
        print(f"\nAction Dim: {len(act_min)}")
        print(f"Action Min: {np.round(act_min, 4)}")
        print(f"Action Max: {np.round(act_max, 4)}")
    if state_min is not None:
        print(f"\nState Dim: {len(state_min)}")
        print(f"State Min:  {np.round(state_min, 4)}")
        print(f"State Max:  {np.round(state_max, 4)}")

def main():
    parser = argparse.ArgumentParser(description='Calculate HDF5 dataset statistics (State & Action)')
    # TODO
    parser.add_argument('--root_dir', type=str, nargs='+', 
                        default=["/home/lab347-no10/Tianyi/RoboTwin/data"],
                        help='Root directory or directories containing task folders (e.g. --root_dir dir1 dir2)')
    
    parser.add_argument('--output_path', type=str, 
                        default="./stat-500-all.json",
                        help='Output JSON file path')
    
    parser.add_argument('--outlier_path', type=str, 
                        default="./outlier_files.txt",
                        help='Output text file for outliers')
    
    parser.add_argument('--num_processes', type=int, default=16,
                        help='Number of parallel processes')
    
    # TODO
    parser.add_argument('--data_mode', type=str, default='both', 
                        choices=['clean', 'randomized', 'both'],
                        help='Data split to process: clean, randomized, or both')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    collect_stats_multiprocess(
        args.root_dir, 
        args.output_path, 
        args.outlier_path, 
        args.num_processes,
        data_mode=args.data_mode  # 传入参数
    )

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()