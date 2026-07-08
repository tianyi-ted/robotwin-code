import os
import torch
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from omegaconf import OmegaConf
from collections import deque
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# 引入项目依赖
from models.model_runner import ModelFactory, VLAWrapperTwoStage, VLAWrapper
from transformers import AutoProcessor
from utils.vlm_utils import preprocess_vlm_messages

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class RobotWinInference:
    """
    RobotWin VLA 推理类
    功能：
    1. 加载训练好的 Checkpoint (Stage 1 或 Stage 2)
    2. 处理环境 Observation (State History + Image)
    3. 执行 Flow Matching 采样生成动作
    """
    def __init__(
        self, 
        config_path: str, 
        checkpoint_path: str, 
        norm_stats_path: str,
        two_stage_mode: bool,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        smooth_actions: bool = True,  # 是否开启平滑
        smooth_sigma: float = 1.0     # 高斯滤波器的标准差
    ):
        self.device = device
        self.dtype = dtype
        self.two_stage_mode = two_stage_mode
        self.smooth_actions = smooth_actions
        self.smooth_sigma = smooth_sigma
        
        # 动作队列: 用于存储当前 Chunk 中待执行的动作
        self.action_queue = deque()
        # 1. 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        self.config = OmegaConf.load(config_path)
        
        
        self.num_inference_steps = self.config.common.num_inference_steps     # Flow Matching 步数
        self.action_execution_horizon = self.config.common.action_execution_horizon
        
        time_sampler = self.config.training.time_sampler
        # 2. 初始化 VLM Processor
        vlm_ckpt = self.config.model.vlm.checkpoint_path
        logger.info(f"Loading VLM Processor form {vlm_ckpt}...")

        self.vlm_processor = AutoProcessor.from_pretrained(vlm_ckpt, trust_remote_code=False)
        
        # 3. 加载模型 (Model Factory)
        logger.info(f"Loading Model from {checkpoint_path}...")
        

        vlm_model = ModelFactory.create_vlm(self.config.model.vlm.checkpoint_path, dtype, device)
        vlm_hidden_size = vlm_model.config.text_config.hidden_size
        
        
        # =========================================================
        # 根据 two_stage_mode 决定加载哪种网络结构
        # =========================================================
        if self.two_stage_mode:
            logger.info("Initializing Two-Stage Action Model structure...")
            action_model = ModelFactory.create_action_model_two_stage(
                self.config,
                vlm_hidden_size=vlm_hidden_size,
                stage=2,
                stage1_ckpt_path=None, 
                freeze_stage1=False    
            )
            
            self.model = VLAWrapperTwoStage(
                vlm_model=vlm_model,
                action_model=action_model,
                time_sampler=time_sampler,
                visual_feat_layer=self.config.model.visual_feat_layer,
                vlm_feat_layer=self.config.model.vlm_feat_layer,
                device=device,
                dtype=dtype,
                norm_stats_path=norm_stats_path
            )
        else:
            logger.info("Initializing Single-Stage Action Model structure...")
            action_model = ModelFactory.create_action_model(
                self.config,
                vlm_hidden_size=vlm_hidden_size
            )
            self.model = VLAWrapper(
                vlm_model=vlm_model,
                action_model=action_model,
                time_sampler=time_sampler,
                visual_feat_layer=self.config.model.visual_feat_layer,
                vlm_feat_layer=self.config.model.vlm_feat_layer,
                device=device,
                dtype=dtype,
                norm_stats_path=norm_stats_path
            )
        
        # weights_only=False 解决 OmegaConf 带来的 UnpicklingError
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        # print('state_dict:', state_dict)
        

        msg = self.model.action_model.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded Action Model weights. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")
        
        self.model.eval()
        self.model.to(device, dtype)
        
        indices_config = self.config.dataset.indices_config
        
        # TODO
        self.processor = RobotWinInferenceProcessor(
            vlm_processor=self.vlm_processor,
            indices_config=indices_config,
            camera_name=self.config.dataset.get('camera_names', ['head_camera'])[0], # 默认取第一个相机
            device=device,
            dtype=dtype
        )
        
        logger.info("Inference Engine Ready.")

    def reset(self):
        """重置状态: 清空 Processor 历史 buffer 和 动作队列"""
        self.processor.reset()
        self.action_queue.clear() # 清空动作队列
        
    @torch.no_grad()
    def _predict_chunk(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """
        [内部函数] 执行一次完整的模型推理，生成 Action Chunk (32步)
        (逻辑与之前的 predict 相同，但只负责生成)
        """
        # 1. 预处理
        batch = self.processor.process(observation, instruction)
        
        # 2. Conditioning
        qpos_cond = self.model.normalize_state(batch['state']) 

        vlm_feats = self.model.get_vlm_features(batch['vlm_inputs'], self.config.model.vlm_feat_layer)
        
        # 3. Flow Matching Sampling
        B = 1
        action_len = self.config.common.action_chunk_size
        action_dim = self.config.common.action_dim
        
        # Init Noise
        x_t = torch.randn((B, action_len, action_dim), device=self.device, dtype=self.dtype)
        steps = torch.linspace(0, 1, self.num_inference_steps + 1, device=self.device, dtype=self.dtype)
        
        # ODE Solver
        for i in range(self.num_inference_steps):
            t_curr = steps[i]
            dt = steps[i+1] - t_curr
            t_input = t_curr.unsqueeze(0)
            
            pred_v = self.model.action_model(
                t=t_input,
                noisy_actions=x_t,
                qpos_history=qpos_cond,
                vlm_features=vlm_feats,
                dino_features=None
            )
            x_t = x_t + pred_v * dt
            
        # 4. Denormalize
        action_seq = self.model.denormalize_action(x_t) # [1, 32, 14]
        
        action_np = action_seq[0].float().cpu().numpy()  # [32, 14]
        
        if self.smooth_actions:
            action_np = gaussian_filter1d(action_np, sigma=self.smooth_sigma, axis=0, radius=5)
        return action_np 
    
    
    
    

    def step(self, observation: Dict[str, Any], instruction: str = "") -> np.ndarray:
        """
        [对外接口] 获取当前时刻的动作 (Receding Horizon Control)
        
        逻辑：
        1. 检查动作队列是否为空。
        2. 如果为空 -> 调用模型预测新的 Chunk -> 将前 N 步存入队列。
        3. 如果不为空 -> 直接从队列弹出一个动作。
        """
        # 每次调用都强制更新轻量级的 state buffer，保证本体感受历史始终连续且最新
        self.processor.update_state_buffer(observation)
        
        
        if len(self.action_queue) == 0:
            # 队列空了，执行推理
            # logger.info("Action queue empty, running model inference...")
            
            full_chunk = self._predict_chunk(observation, instruction) # [32, 14]
            
            # 截取前 N 步 (Horizon)
            valid_actions = full_chunk[:self.action_execution_horizon] # [N, 14]
            
            # 存入队列
            for act in valid_actions:
                self.action_queue.append(act)
        
        # 否则，即使 obs 进来了，如果队列有动作，我们也暂时忽略当前的 obs（除非实现 history update）
        # 或者在 _predict_chunk 内部，processor.process 会自动处理 obs 并 pad 历史
        # 由于我们上面的逻辑是 queue 空了才 predict，所以 queue 不为空时，processor 没有收到新的 obs。
        # 这会导致下一次 predict 时，历史信息缺失中间几帧。
        
        # [修正逻辑]：为了保证历史信息的连续性，即使不需要预测，也必须把当前 Observation 喂给 Processor 维护 Buffer
        if len(self.action_queue) > 0:
             pass

        # 弹出队首动作
        action = self.action_queue.popleft()
        
        return action



class RobotWinInferenceProcessor:
    """
    用于 RobotWin2.0 环境推理的实时数据预处理器。
    功能：
    1. 维护 Proprioception (State) 的历史 Buffer。
    2. 将 Environment 的 observation 转换为 VLM 和 Action Model 的输入 Tensor。
    """
    def __init__(
        self, 
        vlm_processor: Any, 
        indices_config: Dict[str, List[int]] = None,
        camera_name: str = 'head_camera',
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.processor = vlm_processor
        self.device = device
        self.dtype = dtype
        self.camera_name = camera_name
        
        # 1. 配置历史长度
        self.state_indices = indices_config['state_indices']
            
        # 计算需要的最大历史长度 (e.g., if indices are [-2, -1, 0], max_len is 3)
        # 我们需要存储的历史长度 = 1 + abs(min_index)
        self.history_len = 1 + abs(min(self.state_indices))
        
        # 2. 初始化 Buffer (用于存储最近 N 帧的 16维 state 向量)
        self.state_buffer = deque(maxlen=self.history_len)
        
    def reset(self):
        """每个 Episode 开始前调用，清空历史 Buffer"""
        self.state_buffer.clear()


    def update_state_buffer(self, observation: Dict[str, Any]):
        """
        仅更新状态缓存，不进行图像和 VLM 预处理
        """
        current_state = self._parse_state_from_obs(observation)
        
        if len(self.state_buffer) == 0:
            # 如果是第一帧，用当前帧填充整个历史 (Padding start)
            for _ in range(self.history_len):
                self.state_buffer.append(current_state)
        else:
            self.state_buffer.append(current_state)


    def _parse_state_from_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        将环境返回的 endpose/joint_action 解析为 16维向量。
        格式: [left_pose(7), left_grip(1), right_pose(7), right_grip(1)]
        """
        # 注意：环境返回的 observation 结构中，proprioception 在 'endpose' 或 'joint_action' 中
        # 根据你的 inspect_observation，这里使用 'endpose'
        endpose = obs['endpose']
        
        l_pose = np.array(endpose['left_endpose'], dtype=np.float32)
        l_grip = np.array([endpose['left_gripper']], dtype=np.float32)
        r_pose = np.array(endpose['right_endpose'], dtype=np.float32)
        r_grip = np.array([endpose['right_gripper']], dtype=np.float32)
        
        # Concatenate (16,)
        state_vec = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=0)
        return state_vec

    def process(self, observation: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        """
        处理推理输入。
        注意：假设在调用此方法前，update_state_buffer 已经被调用过。
        """
        # --- 1. 处理 State (Proprioception) ---
        # 从 buffer 获取最新序列
        state_seq_np = np.stack(list(self.state_buffer), axis=0)
        
        # 转换为 Tensor 并增加 Batch 维度 -> [1, T_state, D]
        state_tensor = torch.from_numpy(state_seq_np).to(self.device, self.dtype).unsqueeze(0)
        
        # --- 2. 处理 VLM Input (Image + Text) ---

        # 提取 RGB 图像 (Numpy uint8 [H, W, 3])
        # 注意环境返回的是 observation -> camera_name -> rgb
        if self.camera_name in observation['observation']:
            img_np = observation['observation'][self.camera_name]['rgb']
            
            # 转换为 PIL Image (VLM Processor 需要 PIL 或 List[PIL])
            pil_image = Image.fromarray(img_np)
            
            # 调用 VLM 预处理 (复用 dataloader 的逻辑)
            # preprocess_vlm_messages 通常返回 dict: {'input_ids':..., 'pixel_values':...}
            vlm_inputs_single = preprocess_vlm_messages(instruction, [pil_image], self.processor)
            
            # 手动 Collate: 将单样本 dict 转换为 Batch Tensor
            # 这里的 input_ids 已经是 [1, Seq_Len]，pixel_values 是 [1, 3, H, W] 或 similar
            # 我们需要确保它们都在正确的 device 上
            vlm_inputs = {
                'input_ids': vlm_inputs_single['input_ids'].to(self.device),
                'pixel_values': vlm_inputs_single['pixel_values'].to(self.device, self.dtype),
                'attention_mask': vlm_inputs_single['attention_mask'].to(self.device),
                'image_grid_thw': vlm_inputs_single.get('image_grid_thw').to(self.device) if vlm_inputs_single.get('image_grid_thw') is not None else None,
                'mm_token_type_ids': vlm_inputs_single.get('mm_token_type_ids').to(self.device) if vlm_inputs_single.get('mm_token_type_ids') is not None else None,
            }
        else:
            logger.warning(f"Camera {self.camera_name} not found in observation!")
        
        # --- 4. 组装最终 Batch ---
        batch = {
            'state': state_tensor,          # [1, T, D]
            'vlm_inputs': vlm_inputs,       # Dict with tensors on device
        }
        
        return batch
    
    
class ActionRecorder:
    def __init__(self):
        self.actions = []

    def record(self, action):
        """记录单步的 action，兼容 Tensor 和 Numpy"""
        if hasattr(action, 'cpu'):
            action = action.cpu().detach().numpy()
        # 如果模型返回的是 batch 形式 (1, 14)，去掉 batch 维度
        if action.ndim > 1:
            action = action.squeeze(0)
        self.actions.append(action)

    def plot_and_save(self, save_dir, episode_id):
        """Episode 结束时调用：画图、保存并清空缓存"""
        if not self.actions:
            return
        
        actions_np = np.array(self.actions) # 形状: (T, 14)
        T, D = actions_np.shape
        
        # 创建 7 行 2 列的子图画布 (适合 14 维)
        fig, axes = plt.subplots(7, 2, figsize=(15, 20))
        axes = axes.flatten()
        
        for d in range(min(D, 14)):
            axes[d].plot(actions_np[:, d], color='b')
            axes[d].set_title(f'Action Dimension {d} (Joint Angle)')
            axes[d].set_xlabel('Step')
            axes[d].set_ylabel('Value')
            axes[d].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'action_episode_{episode_id}.png')
        plt.savefig(save_path)
        plt.close(fig)
        
        # 清空列表，准备记录下一个 episode
        self.actions = []
    
    
if __name__ == "__main__":
    # 初始化
    agent = RobotWinInference(
        config_path="./configs/robotwin.yaml",
        checkpoint_path="./checkpoints_vla/stage2/xxx/checkpoint_epoch_25.pt",
        norm_stats_path="./utils/stat-200-10.json",
        two_stage_mode=True
    )
    
    # 模拟环境循环
    # obs = env.reset()
    # agent.reset()
    # for i in range(100):
    #     action = agent.predict(obs, "pick up the apple")
    #     obs, _, _, _ = env.step(action)
    
    
    
    
    