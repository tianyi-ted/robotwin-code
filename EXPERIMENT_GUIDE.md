# RobotWin VLA 实验流程指南

> 本文档记录完整的实验流程，方便后续修改模型超参数后重复实验。

---

## 目录

1. [环境准备](#1-环境准备)
2. [数据准备](#2-数据准备)
3. [配置修改](#3-配置修改)
4. [Stage 1 训练](#4-stage-1-训练-motor-babbling)
5. [Stage 2 训练](#5-stage-2-训练-instruction-following)
6. [推理测试](#6-推理测试)
7. [仿真评估](#7-仿真评估)
8. [实验记录模板](#8-实验记录模板)
9. [常见问题排查](#9-常见问题排查)

---

## 1. 环境准备

### 1.1 激活 Conda 环境

```bash
conda activate RoboTwin
```

确认环境：

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
```

**预期输出**：PyTorch ≥2.4.0，CUDA True，transformers ≥5.13.0

### 1.2 依赖检查

```bash
# 核心依赖
pip list 2>/dev/null | grep -E "torch|transformers|omegaconf|h5py|opencv|tensorboard|scipy|sapien"
```

---

## 2. 数据准备

### 2.1 数据集位置

```
/home/lab347-no10/Tianyi/RoboTwin/data/
└── {task_name}/
    └── demo_clean/
        ├── data/
        │   ├── episode_0.hdf5
        │   ├── episode_1.hdf5
        │   └── ...
        └── instructions/
            ├── episode_0.json
            ├── episode_1.json
            └── ...
```

### 2.2 计算归一化统计量

> **注意**：每次更换数据集后**必须**重新计算，否则归一化错误会导致训练不收敛。

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python utils/calc_stat_analysis.py \
  --root_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --output_path ./utils/stat.json \
  --outlier_path ./utils/outlier_files.txt \
  --num_processes 16 \
  --data_mode clean
```

> 如果数据有 `demo_randomized` 分片，将 `--data_mode` 改为 `both`。

### 2.3 数据概览

```bash
# 查看有多少个任务
ls /home/lab347-no10/Tianyi/RoboTwin/data/

# 查看某个任务有多少 episode
ls /home/lab347-no10/Tianyi/RoboTwin/data/{task_name}/demo_clean/data/ | wc -l

# 查看总帧数
cd /home/lab347-no10/Tianyi/robotwin_code
python -c "
import h5py, glob, numpy as np
files = glob.glob('/home/lab347-no10/Tianyi/RoboTwin/data/*/demo_clean/data/*.hdf5')
total = 0
for f in files:
    with h5py.File(f, 'r') as h:
        total += h['joint_action']['vector'].shape[0]
print(f'Total episodes: {len(files)}, Total frames: {total}')
"
```

---

## 3. 配置修改

所有超参数集中在 `configs/robotwin.yaml`。实验前根据需求修改。

### 3.1 常见修改项速查

| 修改目标 | 参数路径 | 参考值 |
|---------|---------|--------|
| **减小模型容量** | `model.action_expert.depth` | 12 → 6 |
| | `model.action_expert.hidden_size` | 512 → 256 |
| | `model.action_expert.vlm_adapter_num_queries` | 64 → 32 |
| **降低学习率** | `training.learning_rate` | 2e-4 → 5e-5 |
| | `training.learning_rate_finetune_stage1` | 2e-4 → 5e-5 |
| **调整 Batch Size** | `training.batch_size` | 128（根据显存调整） |
| **调整推理精度** | `common.num_inference_steps` | 10 → 50 |
| | `common.action_execution_horizon` | 16 |
| **启用 EMA** | `model.ema.enabled` | false → true |
| **启用图像增强** | `dataset.image_aug` | false → true |
| **切换时间采样** | `training.time_sampler` | logit_normal → uniform |
| **更换数据集** | `dataset.dataset_dir` | 数据集路径 |
| **更换 VLM 模型** | `model.vlm.checkpoint_path` | 模型路径 |

### 3.2 设置 `num_workers`

根据 CPU 核数调整，避免 `UserWarning`：

```yaml
system:
  num_workers: 12  # 不超过 CPU 核数
```

---

## 4. Stage 1 训练 (Motor Babbling)

### 4.1 标准命令

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python train_vla.py \
  --config ./configs/robotwin.yaml \
  --stage1 True \
  --norm_stats_path ./utils/stat.json \
  --save_dir ./checkpoints_vla \
  --epochs 25
```

### 4.2 增加训练轮数

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --stage1 True \
  --norm_stats_path ./utils/stat.json \
  --save_dir ./checkpoints_vla \
  --epochs 200
```

### 4.3 预期输出

```
...
INFO - [Action Model (Trainable)] Stats:
INFO -   - Total Parameters: 57.79M
INFO -   - Trainable Parameters: 57.79M
INFO - VLAWrapperTwoStage initialized in Stage 1 (No VLM) mode.
...
INFO - Checkpoints will be saved to: ./checkpoints_vla/stage1/{timestamp}/
...
Epoch [1/25] Step [44/44] Loss: x.xxxx ...
Epoch 25 Completed. Avg Loss: x.xxxx ...
```

### 4.4 检查训练结果

```bash
# 查看保存的 checkpoint
ls -lh ./checkpoints_vla/stage1/

# 查看 loss 是否收敛（最后一个 epoch 的 loss 应远小于第一个 epoch）
```

---

## 5. Stage 2 训练 (Instruction Following)

### 5.1 基础命令（Stage1 权重未冻结）

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python train_vla.py \
  --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ./checkpoints_vla/stage1/{timestamp}/checkpoint_epoch_{N}.pt \
  --save_dir ./checkpoints_vla \
  --epochs 50
```

### 5.2 冻结 Stage 1 权重（推荐小数据集）

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ./checkpoints_vla/stage1/{timestamp}/checkpoint_epoch_{N}.pt \
  --save_dir ./checkpoints_vla \
  --freeze_stage1 \
  --epochs 200
```

> `--freeze_stage1` 只训练 Cross-Attention 和 Adapter 参数（约 31.55M），适合数据量少的情况。

### 5.3 同时使用降低的学习率

先修改 `configs/robotwin.yaml`：

```yaml
training:
  learning_rate: 5e-5
  learning_rate_finetune_stage1: 5e-5
```

然后运行：

```bash
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ./checkpoints_vla/stage1/{timestamp}/checkpoint_epoch_{N}.pt \
  --save_dir ./checkpoints_vla \
  --freeze_stage1 \
  --epochs 200
```

### 5.4 预期输出

```
...
INFO - Loading Frozen VLM from /home/lab347-no10/Tianyi/robotwin_code/Qwen3-VL-2B-Instruct...
Loading weights: 100%|...| 625/625 [00:00<00:00, ...it/s]
INFO - Converting to Stage 2 Model (Instruction Following)...
INFO - Loading Stage 1 weights from ...
INFO - [Action Model (Trainable)] Stats:
INFO -   - Total Parameters: 89.34M
INFO -   - Trainable Parameters: 89.34M
INFO - VLAWrapperTwoStage initialized in Stage 2 mode.
INFO - Optimizer Grouping:
INFO -   - Group Stage 2 (New params): 111 tensors, LR = 0.0002
INFO -   - Group Stage 1 (Finetune): 131 tensors, LR = 0.0002
...
Epoch 50 Completed. Avg Loss: x.xxxx ...
```

---

## 6. 推理测试

### 6.1 命令行推理测试

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python -c "
from robotwin_infer import RobotWinInference

model = RobotWinInference(
    config_path='./configs/robotwin.yaml',
    checkpoint_path='./checkpoints_vla/stage2/{timestamp}/checkpoint_epoch_{N}.pt',
    norm_stats_path='./utils/stat.json',
    two_stage_mode=True
)
print('Model loaded successfully!')
# 注意：这只是加载测试，实际推理需要 SAPIEN 环境提供 observation
"
```

### 6.2 快速过拟合测试

用训练数据中的某个 episode 测试模型能否重现动作：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python -c "
import h5py, torch, numpy as np
from robotwin_infer import RobotWinInference
from PIL import Image
from transformers import AutoProcessor
from utils.vlm_utils import preprocess_vlm_messages

# 加载模型
model = RobotWinInference(
    config_path='./configs/robotwin.yaml',
    checkpoint_path='./checkpoints_vla/stage2/{timestamp}/checkpoint_epoch_{N}.pt',
    norm_stats_path='./utils/stat.json',
    two_stage_mode=True
)

# 从 HDF5 读取一条轨迹
with h5py.File('/home/lab347-no10/Tianyi/RoboTwin/data/{task_name}/demo_clean/data/episode_0.hdf5', 'r') as f:
    gt_actions = f['joint_action']['vector'][:32]  # 取前32步
    l_pose = f['endpose']['left_endpose'][0]   # 取第一帧状态
    l_grip = f['endpose']['left_gripper'][0]
    r_pose = f['endpose']['right_endpose'][0]
    r_grip = f['endpose']['right_gripper'][0]
    
    # 模拟 observation（简化版，仅测试前向能否跑通）
    obs = {
        'endpose': {
            'left_endpose': l_pose,
            'left_gripper': float(l_grip),
            'right_endpose': r_pose,
            'right_gripper': float(r_grip),
        },
        'observation': {
            'head_camera': {
                'rgb': f['observation']['head_camera']['rgb'][0]  # JPEG bytes
            }
        }
    }
    
    # 解码图像（HDF5 存的是 JPEG bytes）
    import cv2
    img = cv2.imdecode(np.frombuffer(obs['observation']['head_camera']['rgb'], dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    obs['observation']['head_camera']['rgb'] = img
    
    model.reset()
    action = model.step(obs, 'beat the block with hammer')
    print(f'Predicted action shape: {action.shape}')
    print(f'First 5 dims: {action[:5]}')
    print(f'GT first 5 dims: {gt_actions[0, :5]}')
"
```

---

## 7. 仿真评估

### 7.1 评估命令

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python eval_vla.py \
  --config_path ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --ckpt_setting stage2/{timestamp} \
  --checkpoint_ep {N} \
  --test_num 50 \
  --instruction_type unseen
```

### 7.2 参数说明

| 参数 | 含义 | 示例 |
|------|------|------|
| `--ckpt_setting` | checkpoint 目录名（相对 `checkpoints_vla/`） | `stage2/2026-07-07_19-51-24` |
| `--checkpoint_ep` | 使用第 N 个 epoch 的 checkpoint | `25`, `50`, `200` |
| `--test_num` | 测试 episode 数量 | `50` |
| `--instruction_type` | 指令类型: `seen` (训练时见过的) 或 `unseen` (没见过的) | `unseen` |
| `--task_config` | 数据分片 | `demo_clean` 或 `demo_randomized` |

### 7.3 评估不同的 checkpoint

```bash
# 逐个 epoch 测试，看哪个 epoch 效果最好
for ep in 10 20 30 40 50; do
  python eval_vla.py \
    --config_path ./configs/robotwin.yaml \
    --norm_stats_path ./utils/stat.json \
    --ckpt_setting stage2/{timestamp} \
    --checkpoint_ep $ep \
    --test_num 50 \
    --instruction_type unseen 2>&1 | grep -E "Success rate|>>>"
done
```

### 7.4 修改测试任务列表

编辑 `eval_vla.py` 末尾的 `tasks_to_test` 列表：

```python
tasks_to_test = [
    "beat_block_hammer",  # 只测你有的任务
    # 有其他任务就加在这里
]
```

---

## 8. 实验记录模板

每次实验建议创建一个实验日志，记录以下信息：

```markdown
## 实验 {实验编号}

### 基本信息
- 日期: 2026-07-XX
- 数据集: beat_block_hammer (50 episodes, 5751 frames)
- Stage 1 epochs: 25
- Stage 2 epochs: 50

### 超参数
| 参数 | 值 |
|------|-----|
| depth | 12 |
| hidden_size | 512 |
| num_heads | 4 |
| learning_rate | 2e-4 |
| batch_size | 128 |
| freeze_stage1 | False |
| time_sampler | logit_normal |
| num_inference_steps | 10 |
| action_execution_horizon | 16 |
| image_aug | False |
| ema | False |

### 结果
- 成功率: 28% (14/50)
- 测试任务: beat_block_hammer
- 指令类型: unseen

### 备注
- Stage 1 训练时间: ~X 分钟
- Stage 2 训练时间: ~X 小时
- GPU 显存占用: ~X GB
```

---

## 9. 常见问题排查

### 9.1 训练时 OOM (Out of Memory)

```yaml
# 减小 batch size
training:
  batch_size: 64  # 或 32
  batch_size_stage1: 64

# 减小 num_workers
system:
  num_workers: 4
```

### 9.2 DataLoader worker 警告

```
UserWarning: This DataLoader will create 16 worker processes...
Our suggested max number of worker in current system is 12
```

修改 `configs/robotwin.yaml`：

```yaml
system:
  num_workers: 12  # 不超过系统 CPU 核数
```

### 9.3 训练 Loss 不下降

1. 检查归一化统计文件是否正确（`stat.json` 里的 min/max 是否合理）
2. 检查学习率是否太大
3. 检查数据集路径是否正确，HDF5 文件能否正常读取

### 9.4 评估时报 "No Task"

```
SystemExit: No Task
```

`tasks_to_test` 列表中的任务在环境中不存在。只保留你有的任务。

### 9.5 推理时报 KeyError

通常是 Qwen3VL 的 API 兼容性问题。确认：

```bash
python -c "import transformers; print(transformers.__version__)"
```

需要 transformers ≥5.13.0（Qwen3VL 支持）。

---

## 快速参考：完整实验流程

```bash
# 0. 激活环境
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

# 1. 计算归一化统计量（换数据集时必须执行）
python utils/calc_stat_analysis.py \
  --root_dir /home/lab347-no10/Tianyi/RoboTwin/data \
  --output_path ./utils/stat.json \
  --data_mode clean

# 2. Stage 1 训练
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --stage1 True \
  --norm_stats_path ./utils/stat.json \
  --save_dir ./checkpoints_vla \
  --epochs 25

# 3. 记录 Stage 1 checkpoint 路径
# ./checkpoints_vla/stage1/2026-07-XX_XX-XX-XX/checkpoint_epoch_25.pt

# 4. Stage 2 训练
python train_vla.py \
  --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ./checkpoints_vla/stage1/2026-07-XX_XX-XX-XX/checkpoint_epoch_25.pt \
  --save_dir ./checkpoints_vla \
  --epochs 50

# 5. 仿真评估
python eval_vla.py \
  --config_path ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --ckpt_setting stage2/2026-07-XX_XX-XX-XX \
  --checkpoint_ep 50 \
  --test_num 50 \
  --instruction_type unseen
```
