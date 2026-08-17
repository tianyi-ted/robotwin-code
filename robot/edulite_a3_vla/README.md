# EDULITE-A3 VLA 真实机械臂执行

本目录把 `place_red_block_into_box` 的 Stage 2 模型接到数据采集时的
EDULITE-A3 从臂（`can0`）。程序只使用模型输出的前 7 维：

```text
action[0:6] = 从臂 L1～L6 目标关节角（rad）
action[6]   = 夹爪归一化目标，0=闭合，1=打开
action[7:14] 始终忽略（训练数据中的合成右臂零填充）
```

输入 state 与训练完全一致：左侧末端位姿 7 维、夹爪 1 维、右侧 8 维全零；
相机输入仍为 `head_camera` RGB。

## 1. 静态检查（不会访问相机或 CAN）

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_vla.yaml \
  --check-only
```

看到 `STATIC CHECK PASSED` 后才能继续。

## 2. 第一次真实测试

清空工作区，只保留红色方块和盒子；红色方块、盒子、相机、机械臂底座位置应
与采集数据时一致。操作人员必须一直握住物理急停。

当前 30 轮数据中的物体和盒子位置变化很小，因此第一次测试不要主动改变布局、
光照或相机位置。程序使能时会保持 L7 当前开度；数据首帧夹爪开度中位数约为
0.60（0=闭合，1=打开），建议在电机仍失能时先把夹爪放到大致半开状态。

```bash
conda activate RoboTwin
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_vla/real_vla.py \
  --config robot/edulite_a3_vla/config.real_vla.yaml \
  --execute
```

启动阶段只加载模型、打开相机并连接 `can0`，不会使能电机。按键顺序：

1. `a`：使能从臂并低速回到训练初始位 `[0, 5, -5, 0, 0, 0]°`。
2. `p`：机械臂保持不动，仅推理一个 32 步动作块并做数值安全检查。
3. 确认出现 `PREVIEW PASSED` 后按 `r`；等待 3 秒后开始执行。
4. `s`：停止模型动作并保持当前位置。
5. `h`：当前轮停止后，低速回到下一轮的 L1～L6 初始位置；L7 保持当前开度。
6. `e`：立即软件急停并失能电机。该按键由独立线程读取，在 GPU 推理期间也有效。
7. `q`：退出；退出时同样失能电机。

软件急停在当前进程内保持锁存；按下 `e` 后不能再次按 `a` 使能，必须检查现场并
重新启动程序。这可以避免误触按键后立即重新上电。

一轮的推荐循环是：`p → r → s → h → p → r`。如果已经达到 25 秒自动停止，
可以直接按 `h`。`h` 会先停止动作预取和运行控制线程，再独占执行回位；回位后会
恢复安全保持线程，并强制要求重新 `p`，不会复用上一轮的动作块。

## 3. 已启用的保护

- 模型原始输出必须有限且满足 14 维格式；明显越过硬限位会直接急停。
- L1～L6 同时裁剪到机械软限位与本次 30 轮训练数据的动作范围。
- 每次模型目标变化再次限制，并经过二阶临界阻尼、速度和加速度限制。
- 100 Hz 独立控制线程持续检查 CAN 反馈时间戳、电机故障和跟随误差。
- 模型推理延迟时保持最后安全目标，不外推、不追赶补发旧动作。
- 首次执行使用已经通过 `p` 检查的动作块；剩余 8 帧时异步预取下一块，避免周期性停顿。
- 夹爪沿用采集配置的 0.10 Nm/0.30 A 电机限幅和软/硬/停转锁存保护。
- 单轮最长 25 秒，到时自动停止并保持当前位置。
- `SIGINT`、`SIGTERM`、异常退出和 `q` 都会执行失能。

## 4. 当前模型与统计量

```text
Stage 2 checkpoint:
checkpoints_vla/place_block_30/stage2/2026-07-31_15-27-43/checkpoint_epoch_400.pt

Normalization:
utils/stat-edulite-pick-30.json

Instruction:
pick up the red block and place it in the box
```

不要替换 checkpoint 而继续沿用旧统计文件。不同数据集训练出的 checkpoint 必须与
它训练时使用的 `config.yaml` 和 normalization JSON 成套更新。

## 5. 推理可视化与偏差诊断

可视化默认启用，保存位置：

```text
robot/edulite_a3_vla/inference_outputs/
```

按 `p` 后会创建 `preview_时间/`：

```text
model_input_rgb.jpg    模型实际看到的RGB图像
preview_summary.png    关节、夹爪和3D末端预测轨迹
preview_actions.csv    原始14维输出、安全处理后7维输出和FK坐标
```

按 `r` 后会创建 `run_时间/`，按 `s`、达到40秒、急停或退出时完成写入：

```text
execution_overlay.mp4  相机画面及目标/反馈关节、夹爪、力矩叠加
execution_summary.png  原始预测、安全目标、滤波命令、真实反馈对比
execution_trace.csv    每一控制帧的完整数值
result.txt             停止原因、帧数和时长
```

判断偏差来源：

- `model_input_rgb.jpg` 中方块位置就不对：相机、摆放或光照与训练不一致。
- `raw model` 与 `safe target` 差异大：模型输出经常超出训练/机械范围，被裁剪。
- `safe target` 与 `filtered command` 差异大：目标跳变过快，受到速度/加速度保护。
- `filtered command` 与 `feedback` 差异大：机械跟随、负载或控制参数问题。
- 四条曲线基本重合但仍抓偏：主要是模型视觉定位或30轮固定布局数据的泛化问题。

L1～L6 使用 100 Hz 临界阻尼二阶滤波。L7 独立使用与采集程序一致的
目标调理：`alpha=0.45` 的 EMA、`1.0 norm/s` 最大目标变化率和 `0.005`
发送死区。录像叠字中的 `grip raw/cmd/actual` 分别表示模型安全限幅后的原始
意图、L7 滤波后真正发送的 PP 目标和电机反馈。力矩/停转保护触发锁存时，
L7 滤波状态会同步重置到保持位置，避免解除锁存后残留的闭合趋势再次收紧。

模型只输出关节动作，没有显式二维抓取点，因此图中给出的是经过FK得到的三维末端
轨迹，而不是伪造一个图像抓取点。
