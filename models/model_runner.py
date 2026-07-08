import torch
import torch.nn as nn
import logging
import json
from transformers import Qwen3VLForConditionalGeneration


from .vla_model_fm import FlowMatchingDiT, calc_flow_matching_loss
from .vla_model_fm import VLAStage1, VLAStage2

logger = logging.getLogger(__name__)

class ModelFactory:
    """负责初始化 VLM 和 Action 预测模型"""
    
    @staticmethod
    def create_vlm(checkpoint_path, dtype=torch.bfloat16, device="cuda"):
        """加载冻结的 Qwen VLM"""
        logger.info(f"Loading Frozen VLM from {checkpoint_path}...")
        
        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=False
        )
            
        # TODO
        vlm_model.lm_head = nn.Identity()
        
        # 冻结 VLM 所有参数
        vlm_model.eval()
        for param in vlm_model.parameters():
            param.requires_grad = False
        
        return vlm_model

    @staticmethod
    def create_action_model(config, vlm_hidden_size):
        """基于 vla_model_fm 创建 FlowMatchingDiT"""
        logger.info("Initializing FlowMatchingDiT Action Model...")
        model = FlowMatchingDiT(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            vlm_feat_dim=vlm_hidden_size,       # Qwen3 的隐藏层维度 (e.g. 2048)
            hidden_dim=config.model.action_expert.hidden_size, # 从配置读取 hidden_dim (e.g. 384 or 512)
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len if hasattr(config.common, 'proprio_len') else 1, # 也就是 initial_state
            depth=config.model.action_expert.depth,
            num_heads=config.model.action_expert.num_heads,
            vlm_num_queries=config.model.action_expert.vlm_adapter_num_queries,
            use_dino=config.model.use_vision_encoder_feat  
        )
        return model


    @staticmethod
    def create_action_model_two_stage(
            config, 
            vlm_hidden_size=None, 
            stage=1, 
            stage1_ckpt_path=None,
            freeze_stage1=True
            ):
        """
        基于 vla_model_fm 创建 Stage 1 或 Stage 2 模型
        """
        logger.info(f"Initializing Action Model for Stage {stage}...")
        
        # 1. 先创建基础的 Stage 1 模型 (无论是哪个阶段都需要)
        model_stage1 = VLAStage1(
            action_dim=config.common.action_dim,
            proprio_dim=config.common.state_dim,
            hidden_dim=config.model.action_expert.hidden_size,
            action_len=config.common.action_chunk_size,
            proprio_len=config.common.proprio_len if hasattr(config.common, 'proprio_len') else 1,
            num_registers=config.model.action_expert.get('num_registers', 2),
            depth=config.model.action_expert.depth,
            num_heads=config.model.action_expert.num_heads
        )

        if stage == 1:
            logger.info("Returning Stage 1 Model (Motor Babbling)...")
            return model_stage1

        elif stage == 2:
            logger.info("Converting to Stage 2 Model (Instruction Following)...")
            
            # 加载 Stage 1 权重
            if stage1_ckpt_path:
                logger.info(f"Loading Stage 1 weights from {stage1_ckpt_path}")
                checkpoint = torch.load(stage1_ckpt_path, map_location='cpu', weights_only=False)
                state_dict = checkpoint['model_state_dict']
                model_stage1.load_state_dict(state_dict)
            else:
                logger.warning("No Stage 1 checkpoint provided for Stage 2 initialization! Using random initialization.")
            
            # 根据 freeze_stage1 参数决定是否冻结
            if freeze_stage1:
                logger.info("Freezing Stage 1 parameters.")
                for param in model_stage1.parameters():
                    param.requires_grad = False
            else:
                logger.info("Unfreezing Stage 1 parameters for finetuning.")
                for param in model_stage1.parameters():
                    param.requires_grad = True 
                
            model_stage2 = VLAStage2(
                stage1_model=model_stage1,
                vlm_feat_dim=vlm_hidden_size
            )
            return model_stage2
        
        else:
            raise ValueError(f"Unknown stage: {stage}")


class VLAWrapper(nn.Module):
    """
    VLA 包装器：
    1. 接收 batch 数据
    2. 通过 Frozen VLM 提取特征
    3. 将特征传入 FlowMatchingDiT 计算 Loss
    """
    def __init__(self,
                 vlm_model, 
                 action_model, 
                 time_sampler, 
                 visual_feat_layer,
                 vlm_feat_layer,
                 device, 
                 dtype, 
                 norm_stats_path
                 ):
        super().__init__()
        self.vlm = vlm_model
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.visual_feat_layer = visual_feat_layer
        self.vlm_feat_layer = vlm_feat_layer
        
        self.device = device
        self.dtype = dtype
        
        # 加载归一化参数
        self.load_norm_stats(norm_stats_path)
        
        
        if self.vlm is not None:
            # 用于存储截获的 Vision Blocks 输出
            self._visual_blocks_output = None
            
            # 注册 Forward Hook
            last_block = self.vlm.model.visual.blocks[self.visual_feat_layer]
            last_block.register_forward_hook(self._visual_hook_fn)
            


    def load_norm_stats(self, path):
        """
        读取 JSON 文件，加载 robotwin_hdf5 的 action 和 state 的 min/max 用于归一化
        """
        logger.info(f"Loading normalization stats from {path}...")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # 1. 读取 robotwin_hdf5 项 (对应新脚本生成的 key)
            stats = data.get('robotwin2')

            # 分开读取 action 和 state
            action_stats = stats.get('action')
            state_stats = stats.get('state')

            # 2. 注册 Action 统计量
            if action_stats:
                act_min = torch.tensor(action_stats['min'], dtype=torch.float32)
                act_max = torch.tensor(action_stats['max'], dtype=torch.float32)
                self.register_buffer('action_min', act_min)
                self.register_buffer('action_max', act_max)
                logger.info(f"Loaded Action stats - Dim: {len(action_stats['min'])}")
            
            # 3. 注册 State 统计量
            if state_stats:
                state_min = torch.tensor(state_stats['min'], dtype=torch.float32)
                state_max = torch.tensor(state_stats['max'], dtype=torch.float32)
                self.register_buffer('state_min', state_min)
                self.register_buffer('state_max', state_max)
                logger.info(f"Loaded State stats - Dim: {len(state_stats['min'])}")
            
        except Exception as e:
            logger.error(f"Failed to load normalization stats: {e}")
            raise e

    def _visual_hook_fn(self, module, input, output):
        # output 是最后一个 block 的输出 (hidden_states)
        # 在 Qwen3VL 中，VisionBlock 的输出通常是 hidden_states
        # 我们将其暂存起来
        self._visual_blocks_output = output


    def get_vlm_features(self, vlm_inputs, vlm_feat_layer):
        """
        参考 test_dataloader.py 中的提取逻辑
        提取 VLM 最后一层 Hidden States
        """
        # 确保输入在正确的设备上
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in vlm_inputs.items()}
        
        with torch.no_grad():
            # 1. 获取输入嵌入
            inputs_embeds = self.vlm.get_input_embeddings()(inputs['input_ids'])
            
            # 2. 获取视觉特征
            vision_output = self.vlm.get_image_features(
                inputs['pixel_values'], inputs['image_grid_thw']
            )
            image_embeds = vision_output.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(self.device, self.dtype)
            
            # TODO
            raw_visual_feats = self._visual_blocks_output   # torch.Size([640, 1024])
            # print(f"Captured Raw Visual Feats Shape: {raw_visual_feats.shape}")
            
            
            # 3. 混合视觉和文本嵌入
            image_mask, _ = self.vlm.model.get_placeholder_mask(
                inputs['input_ids'], inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            
            # 4. 获取位置编码
            position_ids, _ = self.vlm.model.get_rope_index(
                input_ids=inputs['input_ids'],
                mm_token_type_ids=inputs['mm_token_type_ids'],
                image_grid_thw=inputs['image_grid_thw'],
                video_grid_thw=None,
                attention_mask=inputs['attention_mask']
            )

            # 5. 前向传播提取特征 (Qwen3VL 结构)
            # 注意：这里我们不需要完整的 generate，只需要 forward 得到 hidden_states
            vlm_output = self.vlm.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=inputs['attention_mask'],
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True
            )
            # 取一层特征 [B, Seq_Len, Hidden_Dim]
            vlm_latent_feat = vlm_output.hidden_states[vlm_feat_layer]
            
            # print('vlm_latent_feat:', vlm_latent_feat.shape)  # torch.Size([4, 114, 2048])
            return vlm_latent_feat


    def _normalize_tensor(self, x, min_val, max_val):
        """
        内部通用归一化函数：从 [min, max] -> [-1, 1]
        """
        # 强制对齐设备和精度
        min_v = min_val.to(device=x.device, dtype=x.dtype)
        max_v = max_val.to(device=x.device, dtype=x.dtype)
        
        # 防止除以 0
        denominator = max_v - min_v
        denominator[denominator < 1e-6] = 1.0 
        
        # 归一化公式
        norm_x = 2 * (x - min_v) / denominator - 1
        return norm_x

    def normalize_action(self, action):
        """ 专门用于 Action 的归一化 """
        return self._normalize_tensor(action, self.action_min, self.action_max)

    def normalize_state(self, state):
        """ 专门用于 State (Proprioception) 的归一化 """
        return self._normalize_tensor(state, self.state_min, self.state_max)

    def denormalize_action(self, norm_action):
        """ 
        反归一化：用于推理时将模型输出的 [-1, 1] 还原为真实动作 
        公式: x = (norm + 1) / 2 * (max - min) + min
        """
        action_min = self.action_min.to(device=norm_action.device, dtype=norm_action.dtype)
        action_max = self.action_max.to(device=norm_action.device, dtype=norm_action.dtype)
        
        denominator = action_max - action_min
        denominator[denominator < 1e-6] = 1.0
        
        action = (norm_action + 1) / 2 * denominator + action_min
        return action
    
    

    def forward(self, batch):
        """
        前向传播并计算 Loss
        """
        # 1. 提取 VLM 特征 (Frozen)
        vlm_feats = self.get_vlm_features(batch['vlm_inputs'], self.vlm_feat_layer)
        
        # 2. 准备 Action 模型输入
        # x1: 真实的动作序列 [B, T, D]
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        
        # qpos: 机器人当前状态/本体感知 [B, D] -> [B, 1, D]
        qpos_raw = batch['state'].to(self.device, self.dtype)
        
        
        if qpos_raw.dim() == 2: 
            qpos_raw = qpos_raw.unsqueeze(1) 
        
        
        # ==========================================
        # 3. 归一化 (Normalize) 到 [-1, 1]   分别调用对应的归一化函数
        # ==========================================
        x1 = self.normalize_action(x1_raw)  # 使用 action_min/max (14维)
        qpos = self.normalize_state(qpos_raw) # 使用 state_min/max (16维)
        
        
        # 3. 计算 Flow Matching Loss
        loss, _ = calc_flow_matching_loss(
            self.action_model, 
            x1=x1, 
            vlm_features=vlm_feats, 
            qpos_history=qpos,
            time_sampler=self.time_sampler
        )
        
        return loss


class VLAWrapperTwoStage(VLAWrapper):
    """
    两阶段训练专用的 VLA Wrapper。
    继承自 VLAWrapper 以复用归一化(normalization)和辅助函数。
    
    特性:
    1. Stage 1 (Motor Babbling): vlm_model 为 None。跳过 VLM 特征提取，直接训练 Action Model。
    2. Stage 2 (Instruction Following): vlm_model 存在。提取 VLM 特征并传入 Action Model。
    """

    def __init__(self,
                 vlm_model, 
                 action_model, 
                 time_sampler,
                 visual_feat_layer,
                 vlm_feat_layer,
                 device,
                 dtype, 
                 norm_stats_path
                 ):
        # 不直接调用 super().__init__，因为父类可能会在 vlm_model 为 None 时报错 (例如注册 hook 时)
        nn.Module.__init__(self)
        
        self.vlm = vlm_model
        self.action_model = action_model
        self.time_sampler = time_sampler
        self.visual_feat_layer = visual_feat_layer
        self.vlm_feat_layer = vlm_feat_layer
        self.device = device
        self.dtype = dtype
        
        # 判断当前阶段
        self.stage1_mode = (self.vlm is None)
        logger.info(f"VLAWrapperTwoStage initialized in {'Stage 1 (No VLM)' if self.stage1_mode else 'Stage 2'} mode.")

        # 复用父类的加载归一化参数逻辑
        self.load_norm_stats(norm_stats_path)
        
        # 仅在 Stage 2 (有 VLM) 时注册 Hook
        if not self.stage1_mode:
            self._visual_blocks_output = None
            # 注册 Forward Hook 到最后一个 Visual Block
            # 注意：需确保 vlm 模型结构与 Qwen3VL 一致
            last_block = self.vlm.model.visual.blocks[self.visual_feat_layer]
            last_block.register_forward_hook(self._visual_hook_fn)
            logger.info("Registered forward hook on VLM visual blocks.")

    def forward(self, batch):
        """
        前向传播逻辑：
        Stage 1: 读取 Action/State -> 归一化 -> 计算 Flow Matching Loss (vlm_feat=None)
        Stage 2: 读取 VLM Inputs -> 提取特征 -> 读取 Action/State -> 归一化 -> 计算 Loss
        """
        
        # 1. 提取 VLM 特征 (仅 Stage 2)
        vlm_feats = None
        if not self.stage1_mode:
            # 复用父类的 get_vlm_features 方法
            # 注意：batch['vlm_inputs'] 在 Stage 1 Dataloader 中是 None，这里不会被执行
            vlm_feats = self.get_vlm_features(batch['vlm_inputs'], self.vlm_feat_layer)
        
        # 2. 准备 Action 模型输入
        # x1: 真实的动作序列 [B, T, D]
        x1_raw = batch['action_sequence'].to(self.device, self.dtype)
        
        # qpos: 机器人当前状态/本体感知 [B, D] -> [B, 1, D]
        qpos_raw = batch['state'].to(self.device, self.dtype)
        
        # 维度调整: 如果 state 是 [B, 16]，调整为 [B, 1, 16] 以匹配序列输入要求
        if qpos_raw.dim() == 2: 
            qpos_raw = qpos_raw.unsqueeze(1) 
        
        # 3. 归一化 (Normalize) 到 [-1, 1]
        # 直接调用父类 VLAWrapper 的归一化方法
        x1 = self.normalize_action(x1_raw)
        qpos = self.normalize_state(qpos_raw)
        
        
        
        
        
        # 4. 计算 Flow Matching Loss
        # calc_flow_matching_loss 函数内部需要兼容 vlm_features=None 的情况
        # (即在 Stage 1 时，Condition 只有 qpos，没有 visual/lang)
        loss, loss_dict = calc_flow_matching_loss(
            model=self.action_model, 
            x1=x1, 
            vlm_features=vlm_feats, 
            dino_features=None,
            qpos_history=qpos,
            time_sampler=self.time_sampler
        )
        
        return loss

