# EDULITE-A3 主从示教与数据采集操作手册

本文档对应以下程序和配置：

- 采集程序：`robot/edulite_a3_collect/collect.py`
- 主从配置：`robot/edulite_a3_collect/config.master_slave.yaml`
- 数据验证：`robot/edulite_a3_collect/validate_dataset.py`

当前机械臂角色为：

| 角色 | CAN 接口 | 程序配置键 | 用途 |
|---|---|---|---|
| 主臂 | `can1` | `arms.left` | 人手拖动，进入重力补偿零力矩模式 |
| 从臂 | `can0` | `arms.right` | 跟随主臂相对关节位移，执行任务并被记录 |

配置键 `left/right` 在主从模式中表示“主/从角色”，不表示机械臂实际安装在桌面的左右位置。

## 1. 当前控制逻辑

按下 `a` 后，程序执行：

```text
使能两台机械臂
→ 从臂启动 200 Hz 控制循环并保持当前位置
→ 主臂切换到重力补偿零力矩模式
→ 主臂启动 200 Hz 重力补偿循环
→ 等待两臂反馈新鲜且所有电机已使能
→ 记录主、从臂相对零点
→ 主臂 L7 切换到 MOTION_CONTROL，设置 0.10 Nm/0.30 A 上限
→ 50 Hz 发送 Kp=0、Kd=0.05、torque_ff=0 并读取反馈角度
→ 启动 150 Hz 主从映射线程
```

主从关节映射为：

```text
从臂目标 = 从臂启动位置
         + scale × direction × (主臂当前位置 - 主臂启动位置)
```

随后依次经过：

```text
软限位
→ 二阶临界阻尼滤波
→ 速度限制
→ 加速度限制
→ SDK EMA/PD 与硬限位
→ 从臂电机
```

当前 `scale` 为 `1.0`，即主臂与从臂采用 1:1 相对关节位移映射。

主夹爪采用连续绝对开合映射：

```text
主臂 L7 实际角度
→ 按主臂开/闭端点归一化到 [0,1]
→ 50 Hz 一阶滤波、死区和目标变化率限制
→ 从臂归一化夹爪目标
→ 从臂 PP 位置控制及力矩/电流保护
```

`0=完全闭合、1=完全打开`。主臂 L7 在示教期间运行低阻尼 MIT 零力矩运控；
从臂 L7 仍保持 PP 位置控制。
为避免按 `a` 时因主夹爪任意初始角度引起
从夹爪运动，从夹爪最初保持当前位置，必须先把主夹爪明确打开到 `norm>=0.50`
才会激活跟随。

当前响应参数为：

```yaml
sdk:
  smoothing_alpha: 0.65
  max_velocity: 1.0
  max_acceleration: 5.0
master_slave:
  rate_hz: 150
  filter_omega: 18.0
  max_velocity: [0.80, 0.80, 0.80, 1.00, 1.00, 1.00]
  max_acceleration: [4.0, 4.0, 4.0, 5.0, 5.0, 5.0]
```

离线 `0.3 rad` 阶跃测试中，软件滤波器达到目标 95% 的时间由原保守配置
约 `1.73 s` 降至约 `0.49 s`。该结果不包含真实电机、负载和 CAN 延迟。

## 2. 环境准备

```bash
conda activate edulite_collect

cd /home/lab347-no10/Tianyi/robotwin_code
```

检查关键依赖：

```bash
python - <<'PY'
import pinocchio
import cv2
import h5py
import yaml

print("Pinocchio:", pinocchio.__version__)
print("OpenCV:", cv2.__version__)
print("Environment OK")
PY
```

若 SDK 尚未安装：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code/robot/EDULITE_A3/el_a3_sdk
pip install -e ".[dynamics]"
cd /home/lab347-no10/Tianyi/robotwin_code
```

## 3. CAN 接口准备

当前映射：

```text
can1 = 主臂
can0 = 从臂
```

初始化两路 CAN：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code/robot/EDULITE_A3/el_a3_sdk

sudo bash scripts/setup_can.sh can0 1000000
sudo bash scripts/setup_can.sh can1 1000000
```

检查接口：

```bash
ip -details link show can0
ip -details link show can1
```

机械臂上电后可分别观察反馈：

```bash
candump can0
candump can1
```

不要使用来源不明的 `cansend` 指令测试电机。

## 4. 启动前安全检查

每次启动前确认：

1. 主臂确实连接 `can1`，从臂确实连接 `can0`。
2. 两臂均空载或使用已确认安全的轻负载。
3. 从臂周围没有人员、硬障碍物和容易损坏的物体。
4. 两臂均处于远离奇异位姿和机械硬限位的姿态。
5. 物理急停处于操作者随手可按的位置。
6. 摄像头固定牢靠，任务区域完整可见。
7. 正式采集前已标定从臂的 `base_xyz/base_rpy`。
8. 已分别确认主、从臂 L7 的 `closed_rad/open_rad`；不能只假设两台完全一致。
9. 主夹爪进入 MIT 零力矩模式后可用手轻松开合，没有卡滞、异常驱动力或夹手风险。

软件键盘 `e` 只是第二层保护，不能代替物理急停。

## 5. 启动命令

```bash
conda activate edulite_collect
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.master_slave.yaml
```

正常连接时会显示：

```text
Connected. mode=master_slave_single
```

此时只连接了 CAN，尚未进入主从控制。

## 6. 按键功能

| 按键 | 功能 | 说明 |
|---|---|---|
| `a` | 使能并首次回位 | 两臂 J1～J6 回到配置初始位后主动保持；此时不进入示教 |
| `h` | 手动双臂回位 | 仅限未录制时；退出示教并回位，完成后继续等待 `r` |
| `r` | 预约示教与录制 | 先保持初始位倒计时 3 秒，再切换零力矩、建立相对参考并开始采集 |
| `s` | 停止、保存并回位 | 保存成功后退出示教，两臂自动返回初始位并等待下一次 `r` |
| `d` | 丢弃并回位 | 清空本轮缓存，退出示教并回位，不写 HDF5 |
| `e` | 软件急停 | 停止录制、丢弃缓存并失能两台机械臂 |
| `q` | 退出程序 | 未保存缓存会丢弃；退出过程会失能并断开两臂 |
| `[` | 键盘备用关闭 | 每次减少归一化夹爪值；主夹爪控制正常时通常不使用 |
| `]` | 键盘备用打开 | 每次增加归一化夹爪值，并解除力矩停止锁存 |
| `c` | 键盘备用完全关闭 | 发送实测完全关闭角 `70°`，受力矩停止及锁存保护 |
| `o` | 键盘备用完全打开 | 发送实测完全打开角 `-20°`，并解除力矩停止锁存 |
| `t` | 打印夹爪诊断 | 显示 L7 角度、归一化值、扭矩、q 轴电流和保护状态 |
| `Ctrl+C` | 请求退出 | 程序执行失能、停止控制循环和断开 CAN |

按键必须在运行采集程序的终端窗口中输入，不需要按 Enter。

### 主夹爪激活与操作

1. 第一次测试必须空载，不要按 `r`。
2. 按 `a` 后两臂会主动回到 `[0°, 5°, -5°, 0°, 0°, 0°]`。回位期间
   不要手扶或拖动机械臂，并保持运动空间清空。
3. 确认显示 `HOME READY`；此时两臂主动保持位置，不要拖动。
4. 按 `r` 后两臂继续主动保持，终端显示 3 秒非阻塞倒计时。倒计时期间
   `e`/`q` 仍可响应，`s`/`d` 可以取消启动。
5. 倒计时结束后确认 SDK 的
   `夹爪运控示教模式已启用`，随后显示
   `Master L7 is MOTION ZERO-TORQUE PASSIVE`。
6. 此时轻轻移动主夹爪。若仍有明显电机阻力，立即按物理急停，不要硬掰。
7. 将主夹爪打开至 50% 以上（当前标定约为 60°），确认显示
   `MASTER GRIPPER CONTROL ACTIVE`。
8. 从夹爪会按配置的最大目标变化率平滑打开，不会瞬间跳到主夹爪角度。
9. 缓慢闭合和打开主夹爪，确认从夹爪方向及行程一致。
10. 零力矩切换成功后程序才开始保存第一帧；完成后按 `s` 保存并自动退出示教、回位。

当前 episode 回位参数：

```yaml
episode_reset:
  enabled: true
  target_deg: [0.0, 5.0, -5.0, 0.0, 0.0, 0.0]
  home_on_arm: true
  return_after_save: true
  return_after_discard: true
  max_velocity_rad_s: 0.25
  max_acceleration_rad_s2: 0.50
  tolerance_deg: 2.0
  timeout_s: 20.0

master_slave:
  start_delay_s: 3.0
```

回位采用关节空间 S-curve，不包含环境碰撞规划。`s`/`d` 本身就是执行
自动回位的确认动作；按键前应确认两臂路径无人员、物体和线缆阻挡。
L7 在回位过程中保持当前开度，不自动打开或关闭。回位完成后主臂不是
零力矩状态，而是主动保持初始位，直到下一次按 `r`。

当前映射参数：

```yaml
master_gripper_control:
  enabled: true
  activation_open_norm: 0.50
  command_rate_hz: 50
  teaching_kd: 0.05
  motor_torque_limit_nm: 0.10
  motor_current_limit_a: 0.30
  feedback_refresh_rate_hz: 50
  feedback_refresh_timeout_s: 0.04
  position_limit_margin_rad: 0.15
  filter_alpha: 0.45
  deadband_norm: 0.005
  max_target_rate_norm_s: 0.75
  reopen_latch_margin_norm: 0.05
  feedback_timeout_s: 0.30
```

旧 ROS 参考程序对主 L7 使用 `Kp=0、Kd=0.3、torque_ff=0`；当前程序把
`Kd` 降至 `0.05` 以减小手动阻力，并设置 0.10 Nm/0.30 A 电机侧上限。
每次反馈最多等待 0.04 秒，并拒绝超出当前实测标定行程及 0.15 rad 容差的
异常读数。旧 LeRobot 数据中 L7 的零偏不同，不能直接复用其角度端点。

如果 J1～J7 任一失能或反馈陈旧，或者主 L7 的有效零力矩反馈超过
`0.30 s` 未更新，程序会执行
`TELEOP SAFETY STOP`，不会继续
使用不可信的夹爪输入。

按 `t` 时程序通过电机参数 `IQF (0x701A)` 单次读取 L7 的滤波 q 轴电流，
并显示有符号值、绝对值及当前驱动电流上限，例如：

```text
iq=+0.086 A (abs=0.086, limit=0.300 A)
```

IQF 不是母线电流，也不是三相电流的逐相采样值；它适合观察电机负载趋势，
不能用于推算精确的夹爪指尖力。该参数没有放进高频控制循环，只有按 `t` 时才
查询一次，避免增加 CAN 总线负载。如果单次参数查询超时，输出会显示
`iq=unavailable`，但不会因此执行急停。

当前主、从夹爪使用不同零偏：

```yaml
arms:
  left:   # 主夹爪；实机确认编码器角度增大=闭合
    gripper:
      closed_rad: 1.8325957146  # 105°，上位机复核
      open_rad: 0.2617993878    # 15°，上位机复核
  right:  # 从夹爪，当前实测 70° 闭合、-20° 打开
    gripper:
      closed_rad: 1.2217304764
      open_rad: -0.3490658504
```

模型和数据集中的归一化定义保持为 `0=完全关闭、1=完全打开`。`[`/`]`
每次改变总行程的 2%，约等于 L7 电机角 1.8°。

L7 使用 PP 位置模式，当前另有限速：

```yaml
sdk:
  gripper_velocity: 0.55
  gripper_acceleration: 1.00
```

从夹爪完整 90° 行程的恒速理论下限由约 4.49 秒缩短到约 2.86 秒，
实际时间还会受到加速度、PP 控制器和负载影响。即使有限速，`c` 仍会最终
到达机械完全关闭位置；夹持易碎物体时优先使用
`[` 分步闭合。位置限速本身不是力控；下述力矩停止功能也必须标定后才能启用。

因实测闭合冲击仍可能损伤夹爪，当前主从配置已取消启动自动闭合：

```yaml
gripper_control:
  initial_state: hold_current
```

按 `a` 后从臂夹爪保持启动时位置，不会自动打开或闭合。主夹爪达到显式激活
位置后，才由主夹爪连续控制从夹爪。

### 夹爪力矩停止

程序可以在关闭过程中监控 L7 电机反馈扭矩。软件保护分成三路：

- 硬停止：单帧原始扭矩达到硬阈值，立即触发；
- 软停止：夹爪接近停转、仍存在闭合位置误差，并且滤波扭矩连续多帧达到软阈值。
- 停转停止：扭矩未稳定达到软阈值，但夹爪仍有闭合误差，并在具有负载的情况下
  持续接近静止，判定为已经夹到物体。

触发后程序会：

1. 停止继续向关闭方向运动；
2. 将 PP 目标设为检测到的当前位置，继续保持该夹紧位置；
3. 当前配置不再主动向打开方向回退；
4. 锁存关闭保护，之后的 `[` 和 `c` 都不会重新闭合；
5. 在终端打印 `GRIPPER FORCE STOP`。主夹爪必须比从夹爪保持位置明确打开至少
   5%，才会解除锁存；键盘备用的 `]` 或 `o` 也可以解除。

闭合方向现在根据 L7 的实际反馈位置与新目标比较，而不是根据上一次键盘目标
比较。因此，重复按下或长按 `c` 不会再把“正在闭合”状态错误清除。
连续收到相同的 `c` 目标时也不会反复发送 PP 命令或清零接触检测计数。

注意，L7 的单位是电机估算扭矩 Nm，不是夹爪指尖力 N。正式启用前必须
在你的机械结构和物体条件下标定：

1. 保持 `force_stop.enabled: false`；
2. 空载时按 `o` 打开夹爪；
3. 用 `[` 分步关闭，并在多个位置按 `t`，记录空载扭矩最大值；
4. 用不易损坏的测试物体重复操作，在达到期望夹持程度时按 `t`；
5. 阈值应高于空载运动扭矩和噪声峰值，并不高于期望夹持扭矩；
6. 先用软质测试物体、低夹爪速度验证；
7. 确认后修改配置并设置 `enabled: true`。

配置项：

```yaml
gripper_control:
  motor_torque_limit_nm: 0.10
  motor_current_limit_a: 0.30
  force_stop:
    enabled: true
    torque_threshold_nm: 0.12
    hard_torque_limit_nm: 0.18
    ema_alpha: 0.50
    consecutive_samples: 3
    min_closing_time_s: 0.0
    max_contact_velocity_rad_s: 0.05
    min_target_error_norm: 0.015
    stall_time_s: 0.30
    max_stall_velocity_rad_s: 0.01
    stall_torque_threshold_nm: 0.07
    backoff_norm: 0.0
```

该阈值基于 2026-07-23 本机测试：早期空载样本约 `0.004–0.042 Nm`，但后续
完整日志显示空载动态过程也可能出现约 `0.06–0.14 Nm` 的瞬态值，且最终出现
L7 故障码 `16`（A 相过流）。因此软阈值不再单独依赖扭矩，而要求 L7 速度低于
`0.05 rad/s`，用来区分正常动态摩擦和夹爪接触后的停转。

若受到驱动电流限制，实际夹持时扭矩可能一直低于 `0.12 Nm`。停转兜底会在
EMA 扭矩不低于 `0.07 Nm`、速度不高于 `0.01 rad/s` 且持续 `0.30 s` 时触发，
终端显示 `GRIPPER FORCE STOP (STALL)`。它与软/硬停止一样会保持当前位置并
锁存闭合，物体拿走后不会继续向完全关闭位置运动。

现在每次使能后会向 L7 驱动器写入并回读验证：

```text
LIMIT_TORQUE = 0.10 Nm
LIMIT_CUR    = 0.30 A
```

其中 `LIMIT_CUR` 是速度/位置模式明确使用的电流限制。两项均为掉电丢失，程序
每次启动都会重新设置，写入或回读失败时拒绝开始示教。软件阈值、电流限制与
驱动器反馈都不是经过认证的指尖力控系统；空载复测完全通过前不要夹持物体。

出现 L7 故障码 `16` 后，应先停止测试、断电检查夹爪是否发热或卡滞，并按设备
要求清除故障。重新测试时必须空载：

1. 按 `a` 后确认终端打印 `torque=0.100 Nm, current=0.300 A` 的回读结果；
2. 按 `o` 打开夹爪；
3. 只短按一次 `[`，确认正常运动期间不会因软阈值误停止；
4. 再分步闭合；接触机械端点时应只打印一次 `GRIPPER FORCE STOP`；
5. 停止锁存后再次按 `[` 或 `c`，应只打印 `GRIPPER CLOSE BLOCKED` 且不运动；
6. 按 `]` 或 `o` 解除锁存，再确认可以打开；
7. 若仍出现连续停止、异常声响、明显发热或任何 L7 故障码，立即按物理急停，
   不要提高电流/扭矩阈值，也不要继续夹持物体。

## 7. 首次主从运动测试

第一次或修改配置后，先不要录制：

1. 启动程序。
2. 按 `a`。
3. 不移动两臂，等待至少 5 秒。
4. 确认程序保持 `ARMED`，没有安全停止。
5. 缓慢移动主臂 J1 约 2°，观察从臂方向。
6. 依次测试 J2～J6，每次只移动一个关节。
7. 测试结束按 `e`，然后按 `q` 退出。

若某一关节方向相反，立即急停并修改：

```yaml
master_slave:
  direction: [1, 1, 1, 1, 1, 1]
```

例如 J2 方向相反：

```yaml
direction: [1, -1, 1, 1, 1, 1]
```

不要通过快速拖动主臂测试方向。

## 8. 录制一条示教

确认六个关节方向和夹爪均正确后：

1. 将任务物体恢复到标准初态。
2. 把主、从臂移动到安全初始姿态。
3. 启动程序并按 `a`。
4. 等待主从系统稳定。
5. 按 `r` 开始录制。
6. 通过主臂完成一次完整任务。
7. 按 `s` 停止并保存。
8. 失败或不完整的示教按 `d` 丢弃。
9. 恢复场景后再录制下一条。

建议一条 episode 只包含一次完整任务，不要把场景复位过程录入数据。

## 9. 数据输出位置

当前配置：

```yaml
collection:
  output_root: ./collected_data
  task_name: master_slave_test
  split: demo_clean
```

相对路径以配置文件目录为基准，因此实际输出为：

```text
/home/lab347-no10/Tianyi/robotwin_code/robot/edulite_a3_collect/
└── collected_data/
    └── master_slave_test/
        └── demo_clean/
            ├── data/
            │   ├── episode0.hdf5
            │   ├── episode1.hdf5
            │   └── ...
            └── instructions/
                ├── episode0.json
                ├── episode1.json
                └── ...
```

只按 `a` 测试运动不会生成 episode。只有按 `r` 录制并按 `s` 保存后才生成文件。

## 10. 当前数据字段

配置使用：

```yaml
single_arm_padding: zero_right
```

因此输出适配当前模型的 14D action / 16D state 接口：

| HDF5 字段 | 内容 |
|---|---|
| `joint_action/vector[:, 0:6]` | 发给从臂的滤波后六关节目标，单位 rad |
| `joint_action/vector[:, 6]` | 从臂归一化夹爪目标 |
| `joint_action/vector[:, 7:14]` | 全部补零的未使用右侧动作 |
| `endpose/left_endpose` | 从臂末端 `xyz + quaternion(wxyz)` |
| `endpose/left_gripper` | 从臂归一化夹爪反馈 |
| `endpose/right_endpose` | 全零占位 |
| `endpose/right_gripper` | 全零占位 |
| `observation/head_camera/rgb` | JPEG 压缩相机帧 |
| `teleop/master_joint_feedback` | 主臂关节反馈，仅用于诊断 |
| `teleop/slave_joint_feedback` | 从臂关节反馈 |
| `timing/*` | 采样、相机和系统时间戳 |

文件属性包含：

```text
synthetic_zero_right = true
inactive_arm = right
dual_arm_compatible = false
```

这表示数据只是结构上适配双臂网络，实际任务仍是单臂。部署时只能执行模型输出的前 7 维，不能把补零侧输出发送给另一台机械臂。

## 11. 验证采集数据

验证整个任务目录：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code

python robot/edulite_a3_collect/validate_dataset.py \
  robot/edulite_a3_collect/collected_data/master_slave_test \
  --require-dual
```

正常结果类似：

```text
PASS .../episode0.hdf5: T=120 action_dim=14
Checked 1 episode(s), failures=0
```

每次正式批量采集后都应运行验证器。

## 12. 常见安全停止

### 反馈超时

```text
TELEOP SAFETY STOP: master feedback stale/missing ...
```

或：

```text
TELEOP SAFETY STOP: slave feedback stale/missing ...
```

检查 CAN 接口状态、接线、供电、总线负载和反馈频率。不要直接放宽运行期 `feedback_timeout_s`。

### 电机未使能

```text
master has disabled motors: [...]
```

布尔列表对应电机 1～7。检查具体电机故障、模式切换反馈和供电。

### 跟随误差

```text
following error ... exceeds ... rad
```

检查：

- 主从方向是否正确；
- 主臂是否移动过快；
- 从臂是否受阻或负载过大；
- CAN 是否丢帧；
- 控制增益是否合理。

不要优先增大 `max_follow_error_rad`，应先找出无法跟随的原因。

### 初始位置越过硬限位

```text
slave startup joints outside hard limits ...
```

先失能并把从臂移动回合法范围，检查关节零位、方向和 offset 标定。

## 13. 正常结束

无论是否录制，推荐：

1. 按 `e` 失能机械臂；
2. 确认两臂稳定；
3. 按 `q` 退出；
4. 确认日志显示控制循环停止、CAN 接收线程停止和接口断开；
5. 最后再关闭机械臂电源。

如果终端异常关闭，应先使用物理急停确认电机失能，再检查残留进程：

```bash
pgrep -af edulite_a3_collect
```

确认没有采集进程后再重新启动。
