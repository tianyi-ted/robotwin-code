# EDULITE-A3 真实示教采集器

这个目录提供两种采集方式，并把图像、动作、末端状态和语言指令写成当前
`robotwin_code/hdf5_dataloader/dataset.py` 能扫描的目录结构。

## 先明确“两台机械臂”的限制

- `bimanual_kinesthetic`：两台 A3 都作为任务臂，开启 SDK 的重力补偿零力矩模式，操作者直接用手拖动两臂；夹爪用键盘控制。输出动作维度 14、状态维度 16，可用于当前双臂 VLA。这是只有两台机械臂时推荐的方案。
- `master_slave_single`：主从 CAN 角色由所用配置文件决定；当前 `config.master_slave.yaml` 为 `can1` 主臂、`can0` 从臂。主臂相对位移经过二阶临界阻尼滤波、速度/加速度限制、关节限位和跟随误差看门狗后发给从臂；主臂 L7 采用 `Kp=0、低 Kd、torque_ff=0` 的 MIT 零力矩运控并周期反馈角度。原生输出动作维度 7，只能训练单臂版本；也可显式补零适配 14/16 维接口，但这不等于真实双臂数据。
- 真正的“双臂主从”需要四台机械臂：左/右主臂各对应左/右从臂。不要复制一条手臂、补零或把主臂当作第二任务臂；这些做法会让状态和图像中的真实系统不一致。

只做单臂策略原型时，可以显式设置 `collection.single_arm_padding: zero_right`：从臂数据放在模型的“左臂”7维，右 action 7维、右 pose 7维和右 gripper 全部补零。文件会写入 `synthetic_zero_right=true` 和 `inactive_arm=right`，防止被误认成真实双臂数据。该模式只适合部署时完全忽略右侧输出的单臂模型接口适配，不建议与真实双臂 RoboTwin 数据直接混训。

## 输出格式

例如 `output_root=/data/a3`、任务为 `real_pick_and_place` 时：

```text
/data/a3/real_pick_and_place/demo_clean/
├── data/episode0.hdf5
└── instructions/episode0.json
```

双臂 HDF5 的训练字段为：

- `joint_action/vector`: `[T,14]`，顺序为左臂 6 关节、左夹爪、右臂 6 关节、右夹爪；关节单位 rad，夹爪归一化到 `[0,1]`。
- `endpose/{left,right}_endpose`: `[T,7]`，世界坐标的 `xyz + quaternion(wxyz)`。
- `endpose/{left,right}_gripper`: `[T]`，归一化夹爪。
- `observation/head_camera/rgb`: JPEG 字节帧；采集端已处理 OpenCV 通道顺序以适配现有 loader。
- `timing/*` 和 `teleop/*`: 时间戳与诊断数据，不参与当前训练，但用于排查延迟和跟随误差。

在手拖模式中，动作保存当时实际测得的关节位置；现有 dataloader 在每个状态后读取未来 32 帧动作，因此它自然形成示教轨迹。主从模式中动作保存发给从臂的滤波后目标，`teleop/slave_joint_feedback` 另存实际反馈。

## 1. 离线验证程序与数据格式

先在采集用 Python 环境安装依赖。重力补偿和末端 FK 都依赖 Pinocchio，因此真实采集必须安装 SDK 的 `dynamics` extra：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code
pip install -r robot/edulite_a3_collect/requirements.txt
cd robot/EDULITE_A3/el_a3_sdk
pip install -e ".[dynamics]"
cd ../../..
```

先不要连接电机：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code
python robot/edulite_a3_collect/collect.py \
  --config robot/edulite_a3_collect/config.example.yaml \
  --dry-run --auto-seconds 2 \
  --output-root /tmp/a3_dry_run

python robot/edulite_a3_collect/validate_dataset.py \
  /tmp/a3_dry_run --require-dual
```

## 单独预览摄像头

采集器只要求至少一个相机。使用当前 VLA 时保留一个名为 `head_camera` 的固定相机即可，但画面应同时覆盖任务物体、目标区域、夹爪和主要运动范围。单独预览不会打开 CAN，也不会使能电机：

```bash
cd /home/lab347-no10/Tianyi/robotwin_code
python robot/edulite_a3_collect/preview_camera.py \
  --config /path/to/my_a3_collect.yaml \
  --camera head_camera
```

窗口中黄色十字是图像中心，按 `s` 保存截图，按 `q` 或 Esc 退出。临时尝试另一个设备可加 `--source /dev/video2`。应重点确认：整个操作区域没有出画、夹爪闭合点清晰、目标物不被手臂长期遮挡、曝光/焦点稳定，并且相机支架在采集与部署之间不会移动。

当前机器检查到的 `lerobot_a3` 和 `starVLA` 环境安装的是 `opencv-python-headless`，它不能创建 GUI 窗口。可以先不改环境，直接抓取一张带中心十字的实拍图：

```bash
python robot/edulite_a3_collect/preview_camera.py \
  --config /path/to/my_a3_collect.yaml \
  --camera head_camera --no-gui \
  --save-one robot/edulite_a3_collect/camera_snapshots/head_camera.jpg
```

若需要实时窗口，应在具有桌面显示的采集环境中安装带 HighGUI 的 `opencv-python`，并避免让 `opencv-python` 与 `opencv-python-headless` 同时提供同一个 `cv2` 包。

## 2. 真实硬件校准清单

复制配置文件后再修改，原示例中的数值不是你的标定结果：

1. 给两条 CAN 总线配置 1 Mbps，确认 `can0`/`can1` 没有互换，两个机械臂不能共用同一个接口。
2. 单独使用 EDULITE SDK 的低速示例确认每个关节的零位、正方向、软限位和急停。任何一项不一致都应先修改 SDK 的 direction/offset 配置，不能靠采集器掩盖。
3. 实测两侧夹爪的完全闭合角和安全完全张开角，写入 `closed_rad/open_rad`。模型定义是 0=闭合、1=张开；角度正负不影响换算。
4. 标定两机械臂 URDF base 到同一工作台坐标系的 `base_xyz/base_rpy`。错误的外参会让 16 维 proprioception 与图像几何矛盾。
5. 确认 `head_camera` 的视角、分辨率和曝光稳定；训练和部署必须采用同样的相机名称、安装位置和预处理。
6. 检查 Pinocchio、URDF 和惯量参数。采集器会拒绝保存全零 FK 位姿。
7. 物理急停放在操作者随手可按的位置，先在无负载、低速、宽阔区域测试。键盘 `e` 不是物理急停的替代品。
8. 全部核对后才把复制配置中的 `hardware_calibrated` 改为 `true`。

SDK/CAN 的安装和接口初始化仍按 `../EDULITE_A3/el_a3_sdk/README.md` 执行。

## 3. 双臂手拖采集（当前 VLA 推荐）

```bash
cd /home/lab347-no10/Tianyi/robotwin_code
python robot/edulite_a3_collect/collect.py --config /path/to/my_a3_collect.yaml
```

程序启动只连接设备，不会使能。按键：

- `a`：明确使能并进入重力补偿手拖；
- `r`：开始一条 episode；`s`：停止并原子保存；`d`：丢弃；
- `1/2`：左夹爪闭/开，`3/4`：右夹爪闭/开；
- `e`：软件急停并失能；`q`：退出（未保存缓存会丢弃）。

建议每条 episode 从稳定初态开始，只包含一次完整成功任务。失败示范不要混入 clean split；如需研究纠错，应使用单独任务/标签，而不是悄悄混在成功数据里。

## 4. 单臂主从采集

把配置设为 `mode: master_slave_single` 后运行同一命令（临时测试也可加
`--mode master_slave_single`）。当前配置按 `a` 后只让主、从两臂 J1～J6
回到 `[0°, 5°, -5°, 0°, 0°, 0°]` 并主动保持，不会立即进入示教。
按 `r` 后先保持初始位 3 秒，再把主臂切换为零力矩、记录两臂实际角度并
建立相对映射；模式切换成功后才开始录制第一帧。

当前主从配置参考旧 ROS 主从测试，把主臂 L7 设置为 MIT 运控模式，以 50 Hz
发送 `Kp=0、Kd=0.05、torque_ff=0` 并从应答获取角度。安全激活顺序是：

1. 按 `a` 后清空两臂运动空间，等待 `HOME READY`；此时两臂主动保持；
2. 按 `r` 后等待 3 秒倒计时；此时两臂仍主动保持，尚未录制；
3. 倒计时结束后确认终端显示 `Master L7 is MOTION ZERO-TORQUE PASSIVE`；
4. 若进入示教后主夹爪仍有明显驱动力，立即急停；
5. 手动把主夹爪打开到归一化 `0.30` 以上以激活夹爪跟随；
6. 看到 `MASTER GRIPPER CONTROL ACTIVE` 后，主夹爪才开始连续控制从夹爪；
7. 模式切换成功时程序同步开始录制；`s` 保存后退出示教并自动回位；
8. 确实丢弃了一轮数据时，`d` 也会退出示教并自动回位；
9. 未录制时可按 `h` 回位；回位后保持位置，仍需按 `r` 才重新示教；
10. `[`/`]` 和 `c/o` 仅作为键盘备用控制。

倒计时时间由 `master_slave.start_delay_s` 配置；当前为 `3.0` 秒。
主夹爪开度 `0.00～0.80` 线性映射为从夹爪开度 `0.00～1.00`，
主夹爪超过 `0.80` 时从夹爪目标保持为完全打开。

回位使用配置中的 `episode_reset` 速度、加速度、到位容差和超时参数。
它是关节空间轨迹，不做环境碰撞规划；回位期间 L7 保持当前开度。

主夹爪输入经过 50 Hz 限频、低通滤波、死区和目标变化率限制。从夹爪的
电流/扭矩限制及 HARD/SOFT/STALL 保护保持有效。力保护锁存后，必须把主夹爪
明确打开到比从夹爪保持位置大至少 5%，才能恢复跟随。J1～J7 任一失能、
反馈陈旧或 L7 零力矩反馈刷新失败都会触发安全停止。

若确实要用当前 14/16 维网络接口训练“左侧单臂、右侧不用”的策略，可设置：

```yaml
mode: master_slave_single
collection:
  single_arm_padding: zero_right
```

等价的临时命令参数是 `--mode master_slave_single --single-arm-padding zero_right`。这种文件能被当前 dataloader 读取，但当前 loss 没有逐机械臂 mask，右侧零值也会参与训练；推理控制层必须只取左 action `[0:7]`，绝不能把右侧网络输出发给另一台机械臂。

滤波链如下：

```text
主臂反馈 -> 相对零点/方向/比例 -> 软限位 -> 二阶临界阻尼滤波
         -> 速度与加速度限制 -> 从臂 SDK EMA/PD/硬限位 -> 电机
                                  └-> 跟随误差与反馈超时急停
```

第一次实验应进一步降低 `max_velocity/max_acceleration`。若触发跟随误差，先排查方向、零点、负载、增益和通信，不要直接把阈值调大。

## 5. 接入现有训练

只有双臂模式能直接接当前模型。先验证每一条数据：

```bash
python robot/edulite_a3_collect/validate_dataset.py /data/a3 --require-dual
```

然后把 `configs/robotwin.yaml` 的 `dataset.dataset_dir` 指向 `/data/a3`，并保持：

```yaml
common:
  action_dim: 14
  state_dim: 16
dataset:
  camera_names: ["head_camera"]
```

重新计算真实数据统计量，不要继续使用仿真的 `utils/stat.json`。真实相机、关节噪声、机械臂动力学、任务物体和动作频率与仿真存在 domain gap；建议先用 RoboTwin checkpoint 初始化，再以较小学习率微调真实示教，并保留独立真实验证集。部署时还必须把模型输出的归一化动作反归一化、裁剪到相同安全关节范围，并通过独立的实时控制/急停层执行；不能让评测脚本直接绕过安全层发送电机命令。

统计量可用非删除版本生成：

```bash
python utils/calc_stat.py \
  --root_dir /data/a3 \
  --output_path utils/stat_a3_real.json \
  --outlier_path /tmp/a3_outliers.txt \
  --data_mode clean
```

这里不要使用 `calc_stat_remove_outlier.py` 的默认删除逻辑：它把动作绝对值超过 π 当成异常值，但 EDULITE-A3 第 3 关节的合法 SDK 范围可到 `-4.01426 rad`，会误删合法 episode。真实数据的异常判定应使用本配置/SDK 的逐关节限位。

## 主要风险为什么会发生

- **两臂语义错误**：两台设备做主从后，物理任务只有一条从臂，和当前双臂网络的 14D 动作定义不符；补零训练会形成伪相关。
- **坐标/四元数错误**：项目读取的是世界系 `xyz+wxyz`。SDK FK 是基座系 `xyz+rpy`，所以采集器显式组合 base 外参；外参未标定时数值虽然“能训练”，语义却是错的。
- **夹爪标定错误**：仿真和 loader 假定归一化夹爪，直接写电机 rad 会污染统计量，并可能让部署命令超范围。
- **延迟与不同步**：USB 相机、30 Hz 采样和 200 Hz 电机反馈不是硬件同步。程序保存各自时间戳并拒绝陈旧帧，但快速动作仍会有相位差；应保持动作平缓并用 `timing` 检查。
- **主从失稳**：方向错误、启动位置跳变、噪声、过高增益或通信丢包都可能造成突动。相对映射、二阶滤波、速度/加速度/关节限位、反馈超时和连续跟随误差看门狗分别处理这些失效路径，但不能取代物理限位和急停。
- **分布偏移**：仿真 checkpoint 的高成功率不意味着真机安全。尤其是相机外观、执行延迟和接触动力学变化会产生未见过的动作；应从低速、软物体、无障碍、人工随时接管开始。
