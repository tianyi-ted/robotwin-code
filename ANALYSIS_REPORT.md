# RobotWin VLA 项目完整分析报告

> 生成日期: 2026-07-08
> 模型: FlowMatchingDiT + Qwen3VL-2B-Instruct
> 当前成功率: 28%

---

## 一、项目全流程详解

整个系统的数据处理和模型训练流程如下：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              数据流 (Data Flow)                               │
└──────────────────────────────────────────────────────────────────────────────┘

HDF5 文件 (50 episodes, 5751 frames)
│
├─ joint_action/vector  (T, 14)  ← 14维关节角度动作
├─ endpose/                     ← 16维机器人状态
│   ├─ left_endpose   (T, 7)
│   ├─ left_gripper   (T, 1)
│   ├─ right_endpose  (T, 7)
│   └─ right_gripper  (T, 1)
├─ observation/head_camera/rgb  ← JPEG压缩图像
└─ 配套 .json 文件              ← 文本指令 (seen/unseen)

        │
        ▼
┌───────────────── RobotWinTaskDataset (dataset.py) ──────────────────┐
│                                                                      │
│  Stage 1: 预加载 Action+State 到 RAM → 高速随机采样                  │
│  Stage 2: 动态读取 HDF5 + 图像解码 + VLM预处理                      │
│  每次采样: 取当前帧 state[0] + 未来32帧 action[0..31] + 当前帧图像   │
│  文本指令: 从 JSON 中随机选一条                                     │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────── VLAWrapperTwoStage (model_runner.py) ───────────────────┐
│                                                                             │
│  Stage 1 (Motor Babbling):                                                  │
│    Action/State → 归一化 → calc_flow_matching_loss → 更新 VLAStage1 参数    │
│    (无VLM特征，纯动作分布学习)                                              │
│                                                                             │
│  Stage 2 (Instruction Following):                                           │
│    图像+文本 → Qwen3VLProcessor → token IDs → Qwen3VL → hidden_states      │
│    Action/State → 归一化 → concat VLM特征 → calc_flow_matching_loss         │
│    → 更新 VLAStage2 参数 (含 Cross-Attn)                                    │
└──────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────── calc_flow_matching_loss (vla_model_fm.py) ────────────┐
│                                                                     │
│  x₀ = randn(B, 32, 14)     ← 随机噪声                              │
│  x₁ = 真实动作序列          ← target                                │
│  t ~ LogitNormal(μ=0, σ=1) ← 采样时间步                            │
│  xₜ = (1-t)·x₀ + t·x₁     ← 线性插值                              │
│  target_v = x₁ - x₀        ← 目标速度场                            │
│  pred_v = model(t, xₜ, cond) ← DiT预测                             │
│  loss = MSE(pred_v, target_v)                                       │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────── DiT 模型结构 (vla_model_fm.py) ─────────────────┐
│                                                                   │
│  输入: [NoisyAction(32) | Proprio(1) | Registers(N)]             │
│        ↓  + SinCos位置编码                                        │
│  12× DiTBlock:                                                    │
│    ├─ adaLN → Self-Attention (Action↔Proprio↔Registers)           │
│    ├─ adaLN → Cross-Attention (Query=输入, KV=VLM Context)       │
│    └─ adaLN → MLP                                                 │
│        ↓                                                          │
│  Conv1d(k=3) → 输出 (B, 32, 14)  ← 预测的速度场                   │
│                                                                   │
│  VLM Context 生成:                                                 │
│    Qwen3VL hidden_states [-2层] (B, ~556, 2048)                   │
│    → VLMFeatureAdapter (64 queries, 2层 Transformer Decoder)      │
│    → (B, 64, 512)  ← 压缩后的视觉-语言条件特征                    │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌────────────── 推理流程 (robotwin_infer.py) ─────────────────┐
│                                                               │
│  step(obs, instruction):                                      │
│    1. update_state_buffer(obs)   ← 维护状态历史               │
│    2. if 动作队列为空:                                        │
│         _predict_chunk(obs, instruction):                     │
│           - Processor → VLM输入                                │
│           - Qwen3VL → VLM特征                                  │
│           - Flow Matching ODE: t=0→1, 10步                    │
│           - 反归一化 → 动作序列 [32, 14]                      │
│         → 取前16步入队                                        │
│    3. 弹出队首动作 → return                                    │
└───────────────────────────────────────────────────────────────┘
```

---

## 二、每个文件的详细解析

### `configs/robotwin.yaml` — 配置中心

所有超参数的唯一入口。关键参数及含义：

| 参数 | 当前值 | 含义 |
|------|--------|------|
| `action_dim: 14` | 14 | 双臂7+7关节角度 |
| `state_dim: 16` | 16 | 左臂7+左夹爪1+右臂7+右夹爪1 |
| `action_chunk_size: 32` | 32 | 一次预测32步动作序列 |
| `action_execution_horizon: 16` | 16 | 每次只执行前16步 |
| `num_inference_steps: 10` | 10 | ODE求解步数（越多越精确但越慢） |
| `data_mode: "clean"` | clean | 只用demo_clean |
| `model.action_expert.depth: 12` | 12 | DiT Block 层数 |
| `model.action_expert.hidden_size: 512` | 512 | Transformer 隐层维度 |
| `training.batch_size: 128` | 128 | 批量大小 |
| `training.max_steps: 1000000` | 1M | 最大训练步数 |
| `training.learning_rate: 2e-4` | 2e-4 | 学习率 |
| `time_distribution.timestep_sample_method: "logit_normal"` | logit_normal | 时间步采样策略 |

### `hdf5_dataloader/dataset.py` — 数据加载

**关键类和方法**：

- **`RobotWinTaskDataset.__init__`**: 扫描文件夹 → 构建索引映射 → (Stage 1)预加载数据到RAM
- **`_build_index_map()`**: 遍历所有 HDF5 文件，将 50 episodes × ~115帧/episode = 5751 帧展平为全局索引
- **`__getitem__(idx)`**: 
  - Stage 1: 直接从 `buffer_action` / `buffer_state` 内存中切片，极其高效
  - Stage 2: 定位到 episode → 计算局部索引 → 读 HDF5 → 解码 JPEG 图像 → VLM 预处理 → 返回 dict
- **`_load_hdf5_data()`**: 从 HDF5 读取 action/state/图像，图像用 cv2.imdecode 解码 JPEG
- **`_process_vlm_inputs_batch()`**: 对变长的 VLM token 序列做 padding 对齐
- **`collate_fn()`**: 将单样本组装成 batch

### `models/vla_model_fm.py` — 核心模型架构

**关键组件**：

| 类/函数 | 参数 | 说明 |
|---------|------|------|
| `SinusoidalPosEmb` | hidden_dim | 时间步 t → sin/cos 嵌入 |
| `DiTBlockStage1` | hidden=512, heads=4 | 仅 Self-Attn + MLP，adaLN-zero 调制 |
| `VLAStage1` | depth=12, action_dim=14, proprio_dim=16 | Motor Babbling 模型，~57.79M 参数 |
| `DiTBlockStage2Wrapper` | — | 在 Stage1 Block 外套一层 Cross-Attn |
| `VLAStage2` | stage1_model + VLM Adapter | 完整的指令跟随模型，~89.34M 参数 |
| `VLMFeatureAdapter` | 64 queries, 2层 Decoder | 将 Qwen 的 ~556 tokens 压缩为 64 个 |
| `DiTBlock` | — | 完整 Block：Self-Attn + Cross-Attn + MLP |
| `FlowMatchingDiT` | — | 单阶段架构（当前没用到） |
| `calc_flow_matching_loss` | — | 计算 Conditional Flow Matching loss |
| `modulate` | — | adaLN 调制函数 x·(1+scale)+shift |

### `models/model_runner.py` — 模型组装

**关键方法**：

| 方法 | 功能 |
|------|------|
| `ModelFactory.create_vlm()` | 加载 Qwen3VL-2B，移除 lm_head，冻结所有参数 |
| `ModelFactory.create_action_model_two_stage()` | 创建 VLAStage1 或 VLAStage2 |
| `VLAWrapperTwoStage.__init__()` | 初始化 VLM Hook + 归一化参数 |
| `VLAWrapperTwoStage.forward()` | 主前向：提取 VLM 特征 → 归一化 → 计算 Loss |
| `VLAWrapperTwoStage.get_vlm_features()` | 关键方法！通过 Qwen3VL 提取特征 |
| `VLAWrapper.load_norm_stats()` | 从 JSON 加载 state/action 的 min/max |
| `normalize_action/state()` | 将数据缩放到 [-1, 1] |
| `denormalize_action()` | 从 [-1, 1] 还原为真实动作值 |

**VLM 特征提取细节**：
1. 获取 token 嵌入 (`get_input_embeddings`)
2. 获取视觉特征 (`get_image_features` → `pooler_output`)
3. 用 `get_placeholder_mask` 将图像 token 嵌入到文本序列中
4. 获取位置编码 (`get_rope_index`)
5. 语言模型前向 → 取 `hidden_states[-2]` 层输出 → (B, ~556, 2048)

### `train_vla.py` — 训练入口

**Stage 1** (Motor Babbling)：
- 加载配置 → 创建 Dataset（stage1_mode=True，预加载到RAM）
- 创建 VLAStage1 模型（57.79M 参数，全可训练）
- 创建优化器：所有参数在一组，LR=2e-4
- 训练循环：每个 batch 计算 Flow Matching Loss → backward → 梯度裁剪 → optimizer.step
- 每 epoch 保存一次 checkpoint

**Stage 2** (Instruction Following)：
- 加载配置 → 创建 Dataset（stage1_mode=False，需要VLM预处理）
- 加载 Qwen3VL-2B（冻结）→ 创建 VLAStage2（加载 Stage1 权重，89.34M 参数全可训练）
- 创建优化器：两组参数（Stage1 微调 + Stage2 新参数），LR 都是 2e-4
- 训练循环：提取 VLM 特征 → 归一化 → Flow Matching Loss → backward → 梯度裁剪 → optimizer.step
- 每 epoch 保存一次 checkpoint

### `robotwin_infer.py` — 推理引擎

- **`__init__()`**: 加载配置 → 加载 VLM Processor → 加载模型 checkpoint → 创建 Processor
- **`reset()`**: 清空状态缓冲区和动作队列
- **`_predict_chunk()`**: 
  1. Processor 处理 observation + instruction → batch
  2. 归一化 state + 提取 VLM 特征
  3. Flow Matching ODE: 从 x₀~N(0,1) 开始，10步积分到 x₁
  4. 反归一化 → 可选高斯平滑 → [32, 14] 动作序列
- **`step()`**: Receding Horizon Control 滚动执行

### `eval_vla.py` — 仿真评估

- 初始化 SAPIEN 仿真环境 + 加载模型
- 对每个测试任务：
  1. 设置随机种子，生成一条专家演示作为参考
  2. 生成文本指令（从 seen/unseen 中选）
  3. 环境循环：get_obs → model.step → 执行动作 → 检查成功
  4. 记录成功率

---

## 三、28% 成功率的原因分析

### 核心瓶颈：数据量严重不足

```
数据集: beat_block_hammer
  ├─ 50 episodes (演示轨迹)
  ├─ 5751 帧 (总帧数)

模型参数量:
  ├─ VLAStage1: 57.79M 参数
  └─ VLAStage2: 89.34M 参数 (含 Cross-Attn)

数据/参数比: 5751 / 89,340,000 ≈ 0.006%
```

89M 参数的网络只有 5751 帧数据可学，属于极度数据稀疏场景。

### 具体问题

| 问题 | 影响 |
|------|------|
| 数据太少 | 模型严重过拟合，无法泛化到新的场景/位置 |
| 只有1个任务 | 学不到"操作"的通用概念，只是记忆特定轨迹 |
| Stage1+2 都微调 | 89M 参数全部更新，但数据支撑不够 |
| batch_size=128 | 但每个 batch 中大部分是重复的时序帧 |
| 动作平滑(gaussian_filter) | 可能过度平滑导致动作精度下降 |

---

## 四、提升成功率的方案

### 方案 1：增加训练轮数

```bash
# Stage 1: 25 → 200 epochs
python train_vla.py --config ./configs/robotwin.yaml \
  --stage1 True --norm_stats_path ./utils/stat.json \
  --save_dir ./checkpoints_vla --epochs 200

# Stage 2: 50 → 500 epochs (配合冻结 Stage1)
python train_vla.py --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ./checkpoints_vla/stage1/xxx/checkpoint_epoch_200.pt \
  --save_dir ./checkpoints_vla --freeze_stage1 --epochs 500
```

### 方案 2：冻结 Stage 1 + 降低学习率

```yaml
# configs/robotwin.yaml 中修改
training:
  learning_rate: 5e-5     # 从 2e-4 降低
```

```bash
python train_vla.py --config ./configs/robotwin.yaml \
  --norm_stats_path ./utils/stat.json \
  --stage1_checkpoint ... --save_dir ./checkpoints_vla \
  --freeze_stage1 --epochs 200
```

### 方案 3：减少模型容量

```yaml
# configs/robotwin.yaml 中修改
model:
  action_expert:
    depth: 6           # 12 → 6
    hidden_size: 256   # 512 → 256
    vlm_adapter_num_queries: 32  # 64 → 32
```

参数从 89M 降到约 **22M**。

### 方案 4：增加数据多样性（最有效）

- 收集更多演示数据（建议 ≥500 episodes）
- 启用图像增强 `image_aug: true`
- 增加 `demo_randomized` 变体

### 方案 5：启用 EMA + 调整 ODE 步数

```yaml
model:
  ema:
    enabled: true
    update_after_step: 1000
    inv_gamma: 1.0
    power: 0.75
    min_value: 0.0
    max_value: 0.9999
  inference:
    num_inference_timesteps: 50  # 10 → 50
```

### 优先级建议

| 优先级 | 行动 | 预期提升 | 工作量 |
|--------|------|---------|--------|
| 🔴 P0 | 收集更多数据 (≥500 episodes) | +30~50% | 大 |
| 🔴 P0 | 冻结 Stage1 增加 epochs | +5~10% | 小 |
| 🟡 P1 | 减小模型容量 (depth=6, hidden=256) | +10~15% | 中 |
| 🟡 P1 | 降低学习率 (5e-5) | +5~10% | 小 |
| 🟢 P2 | 启用 EMA | +3~5% | 小 |
| 🟢 P2 | 增加 ODE 步数 (10→50) | +2~5% | 小 |
| 🟢 P2 | 启用图像增强 | +2~5% | 小 |
