import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_1d_sincos_pos_embed(embed_dim, length):
    """
    生成标准的 Transformer SinCos 位置编码
    Return: (1, length, embed_dim)
    """
    if embed_dim % 2 != 0:
        raise ValueError("Embed dim must be divisible by 2")

    pos = torch.arange(length, dtype=torch.float32)
    grid = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (grid / (embed_dim // 2)))

    out = torch.einsum('m,d->md', pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.unsqueeze(0)

# --- Time Embedding ---
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = emb.to(dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==========================================
# 2. Stage 1: Motor Babbling (含 Register)
# ==========================================

class DiTBlockStage1(nn.Module):
    """Stage 1 Block: 只有 Self-Attention 和 MLP"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        # 6 params: (shift, scale, gate) * 2
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t):
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = \
            self.adaLN_modulation(emb_t).chunk(6, dim=1)

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        # Stage 1: Self-Attention only
        x = x + gate_msa.unsqueeze(1) * self.attn1(x_norm, x_norm, x_norm)[0]

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x

class VLAStage1(nn.Module):
    def __init__(self, 
                 action_dim=7, 
                 proprio_dim=7, 
                 hidden_dim=384, 
                 num_heads=6, 
                 depth=12, 
                 action_len=32, 
                 proprio_len=8, 
                 num_registers=2
                 ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.num_registers = num_registers
        self.depth = depth
        self.num_heads = num_heads

        # Time Embed
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Projections
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        
        # Positional Embeddings
        self.register_buffer('action_pos_emb', get_1d_sincos_pos_embed(hidden_dim, action_len))
        self.register_buffer('proprio_pos_emb', get_1d_sincos_pos_embed(hidden_dim, proprio_len))
        
        # Register Tokens
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

        # Blocks
        self.blocks = nn.ModuleList([DiTBlockStage1(hidden_dim, num_heads) for _ in range(depth)])
        
        # Output
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(self, t, noisy_actions, qpos_history, vlm_features=None, dino_features=None):
        # Stage 1 显式忽略 vlm 和 dino 特征
        t_emb = self.time_mlp(t)
        
        # 1. Embeddings
        x_action = self.action_proj(noisy_actions) + self.action_pos_emb[:, :noisy_actions.shape[1], :]
        x_proprio = self.proprio_proj(qpos_history) + self.proprio_pos_emb[:, :qpos_history.shape[1], :]
        
        # 2. Concat: [Action, Proprio]
        x = torch.cat([x_action, x_proprio], dim=1)

        # 3. Add Register Tokens
        if self.num_registers > 0:
            regs = self.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x, regs], dim=1)
        
        # 4. DiT Blocks (Self-Attn Only)
        for block in self.blocks:
            x = block(x, t_emb)
            
        # 5. Output
        x = self.final_norm(x)
        # 只取 Action 部分 (切片前 action_len 长度)
        x_action_out = x[:, :self.action_len, :]
        return self.output_proj(x_action_out)

# ==========================================
# 3. Stage 2: Adapter & Cross-Attn
# ==========================================

class DiTBlockStage2Wrapper(nn.Module):
    """Stage 2 Wrapper: 包装 Stage 1 Block，插入 Cross-Attention"""
    def __init__(self, stage1_block, hidden_size, num_heads, dropout=0.):
        super().__init__()
        self.stage1_block = stage1_block # 引用 Stage 1 Block
        
        self.norm_cross = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_cross = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-Attn 专用的 adaLN (3 params: shift, scale, gate)
        self.adaLN_cross = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 3 * hidden_size, bias=True))
        
        # TODO
        nn.init.normal_(self.adaLN_cross[-1].weight, std=1e-4) # 不要用纯0
        nn.init.constant_(self.adaLN_cross[-1].bias, 1e-4)
        
        # nn.init.constant_(self.adaLN_cross[-1].weight, 0)
        # nn.init.constant_(self.adaLN_cross[-1].bias, 0)

    def forward(self, x, emb_t, context):
        # 1. 先进行 Cross-Attention (Stage 2 新增逻辑, 注入 VLM Context)
        (shift_cross, scale_cross, gate_cross) = self.adaLN_cross(emb_t).chunk(3, dim=1)
        x_norm = modulate(self.norm_cross(x), shift_cross, scale_cross)
        x = x + gate_cross.unsqueeze(1) * self.attn_cross(x_norm, context, context)[0]

        # 2. 再进行 Self-Attention 和 MLP (直接完美复用 Stage 1 的 forward 逻辑)
        # 此时的 x 已经融合了 Cross-Attention 的信息
        x = self.stage1_block(x, emb_t)
        
        return x

class VLAStage2(nn.Module):
    def __init__(self, 
                 stage1_model: VLAStage1, 
                 vlm_feat_dim=384, 
                 dino_feat_dim=1024,
                 vlm_num_queries=64,
                 adapter_depth=2,
                 use_dino=False,
                 language_fusion_layers='all'
                 ): # 可以是 "all" 或 list, e.g. [6, 7, 8...]
        super().__init__()
        self.stage1 = stage1_model
        self.use_dino = use_dino
        
        hidden_dim = stage1_model.hidden_dim
        num_heads = stage1_model.num_heads
        
        # 1. 视觉 Token Adapter (压缩 Vision Tokens)
        self.vlm_visual_adapter = VLMFeatureAdapter(
            vlm_feat_dim=vlm_feat_dim,
            hidden_dim=hidden_dim,
            num_queries=vlm_num_queries,
            num_heads=num_heads,
            num_layers=adapter_depth,
            dropout=0.
        )
        
        # 3. DINO Adapter
        if self.use_dino:
            self.dino_adapter = VLMFeatureAdapter(
                vlm_feat_dim=dino_feat_dim,
                hidden_dim=hidden_dim,
                num_queries=vlm_num_queries,
                num_heads=num_heads,
                num_layers=adapter_depth,
                dropout=0.
            )
        
        # 处理 Fusion Layers 逻辑
        if language_fusion_layers == "all":
            self.language_fusion_layers = set(range(stage1_model.depth))
        else:
            self.language_fusion_layers = set(language_fusion_layers)

        # 包装 Blocks
        self.blocks = nn.ModuleList([
            DiTBlockStage2Wrapper(b, hidden_dim, num_heads)
            for b in self.stage1.blocks
        ])

    def forward(self, t, noisy_actions, qpos_history, vlm_features, dino_features=None):
        """
        Stage 2 Forward:
        1. 准备 Input Sequence (Action + Proprio + Registers) [复用 Stage 1]
        2. 准备 VLM Context (Vision Adapter + Lang Proj) [新增]
        3. Cross Attn
        """
        
        # --- Part 1: Embedding & Input Sequence (完全复用 Stage 1 逻辑) ---
        t_emb = self.stage1.time_mlp(t)
        
        x_action = self.stage1.action_proj(noisy_actions) + self.stage1.action_pos_emb[:, :noisy_actions.shape[1], :]
        x_proprio = self.stage1.proprio_proj(qpos_history) + self.stage1.proprio_pos_emb[:, :qpos_history.shape[1], :]
        
        # Concat Action + Proprio
        x = torch.cat([x_action, x_proprio], dim=1)

        # Add Register Tokens (Stage 1 已初始化并训练过，或者被冻结)
        if self.stage1.num_registers > 0:
            regs = self.stage1.register_tokens.expand(x.shape[0], -1, -1)
            x = torch.cat([x, regs], dim=1)
            
        # --- Part 2: 准备 Condition Context (Adapter Logic) ---
        
        # 2.2 处理 Vision (通过 Adapter 压缩)
        
        cond_vlm = self.vlm_visual_adapter(vlm_features)
        c_full_list = [cond_vlm]
        if self.use_dino and dino_features is not None:
            cond_dino = self.dino_adapter(dino_features)
            c_full_list.append(cond_dino) # DINO 放在最后
            
        context_full = torch.cat(c_full_list, dim=1)
        
        # --- Part 3: DiT Block Forward (Wrapped) ---
        for i, block in enumerate(self.blocks):
            if i in self.language_fusion_layers:
                current_context = context_full
            else:
                current_context = context_full
            
            x = block(x, t_emb, current_context)
            
        # --- Part 4: Output (复用 Stage 1) ---
        x = self.stage1.final_norm(x)
        x_action_out = x[:, :self.stage1.action_len, :]
        return self.stage1.output_proj(x_action_out)



class VLMFeatureAdapter(nn.Module):
    def __init__(self, vlm_feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        
        # 特征投影
        self.feature_proj = nn.Linear(vlm_feat_dim, hidden_dim)
        
        # Learnable Queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # Transformer Decoder Layers (包含 Dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,  # 增加随机 dropout 设计
            batch_first=True, 
            activation="gelu",
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, vlm_features):
        """
        vlm_features: (B, N, vlm_feat_dim)
        Returns: (B, num_queries, hidden_dim)
        """
        B = vlm_features.shape[0]
        
        # (B, N, vlm_feat_dim) -> (B, N, hidden_dim)
        memory = self.feature_proj(vlm_features)
        
        # (1, num_queries, hidden_dim) -> (B, num_queries, hidden_dim)
        tgt = self.query_embed.expand(B, -1, -1)
        
        # Decoder forward
        out = self.transformer_decoder(tgt, memory)
        return out


class AdapterDiTBlock(nn.Module):
    """
    支持 Time Conditioning 的 Transformer Decoder Block
    结构：adaLN -> Self-Attn -> adaLN -> Cross-Attn -> adaLN -> FFN
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        # 1. Self-Attention (Queries 之间的交互)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Cross-Attention (Queries 查询 VLM Features)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        # 3. FFN
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # 4. adaLN Modulation (根据 time embedding 生成 shift, scale, gate)
        # 这里的参数量是 6 * hidden_size (Self-Attn) + 3 * hidden_size (MLP) 
        # 我们需要增加 Cross-Attn 的调制参数 -> 9 * hidden_size
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        
        # Zero-init 最后一个线性层，使得初始状态下 block 近似恒等映射
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, context):
        """
        x: (B, num_queries, H) - Queries
        emb_t: (B, H) - Time Embeddings
        context: (B, N, H) - VLM Features (Keys/Values)
        """
        # 生成调制参数
        (shift_msa, scale_msa, gate_msa, 
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(emb_t).chunk(9, dim=1)

        # 1. Self-Attention
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention (Source = context)
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        attn_out, _ = self.attn2(x_norm, context, context)
        x = x + gate_cross.unsqueeze(1) * attn_out

        # 3. MLP
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        
        return x



class VLMFeatureAdapterTAware(nn.Module):
    def __init__(self, vlm_feat_dim, hidden_dim, num_queries=32, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        
        # 特征维度对齐
        self.feature_proj = nn.Linear(vlm_feat_dim, hidden_dim)
        
        # Learnable Queries
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, hidden_dim))
        
        # 使用自定义的 AdapterDiTBlock 替代 nn.TransformerDecoder
        self.blocks = nn.ModuleList([
            AdapterDiTBlock(hidden_dim, num_heads, mlp_ratio=4.0, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, vlm_features, t_emb):
        """
        vlm_features: (B, N, vlm_feat_dim)
        t_emb: (B, hidden_dim)  <-- 新增输入
        """
        B = vlm_features.shape[0]
        
        # 1. 准备 Key/Value (Memory)
        memory = self.feature_proj(vlm_features)
        
        # 2. 准备 Query
        x = self.query_embed.expand(B, -1, -1) # (B, num_queries, H)
        
        # 3. 通过支持 Time Conditioning 的 Block
        for block in self.blocks:
            x = block(x, t_emb, memory)
            
        return x



class FlowMatchingDiT(nn.Module):
    def __init__(
        self, 
        action_dim=7, 
        proprio_dim=7,    
        vlm_feat_dim=384, 
        hidden_dim=384, 
        dino_feat_dim=1024,
        num_heads=6, 
        depth=12, 
        action_len=32, 
        proprio_len=8,
        vlm_num_queries=64,
        adapter_depth=2,
        use_dino=True,  
        adapter_dropout_ratio=0.,
        num_registers=2,
        language_fusion_layers= 'all' # 可以是 "all" 或 list, e.g. [6, 7, 8, 9, 10, 11]
    ):
        super().__init__()
        self.action_len = action_len
        self.proprio_len = proprio_len
        self.use_dino = use_dino
        self.num_registers = num_registers
        self.depth = depth
        
        # 处理层索引配置
        if language_fusion_layers == "all":
            self.language_fusion_layers = set(range(depth))
        else:
            self.language_fusion_layers = set(language_fusion_layers)

        
        # 1. Time Embedding 
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 2. Input Projection (Action)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.register_buffer(
            'action_pos_emb', 
            get_1d_sincos_pos_embed(hidden_dim, action_len)
        )

        # 3. Proprioception (Input for Self-Attention)
        # 本体状态感知特征，现在将拼接到 self-attention 部分
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.register_buffer(
            'proprio_pos_emb', 
            get_1d_sincos_pos_embed(hidden_dim, proprio_len)
        )
        
        # Register Tokens
        # 这些 token 不包含位置编码
        if self.num_registers > 0:
            self.register_tokens = nn.Parameter(torch.randn(1, num_registers, hidden_dim))
            # 使用截断正态分布初始化，利于训练稳定
            nn.init.trunc_normal_(self.register_tokens, std=0.02)
        
        
        
        # 4. Condition Encoders (For Cross-Attention)
        
        # TODO
        # 4.1 视觉 Token Adapter
        self.vlm_visual_adapter = VLMFeatureAdapter(
        # self.vlm_visual_adapter = VLMFeatureAdapterTAware(
            vlm_feat_dim=vlm_feat_dim,
            hidden_dim=hidden_dim,
            num_queries=vlm_num_queries,
            num_heads=num_heads,
            num_layers=adapter_depth,
            dropout=adapter_dropout_ratio
        )
        
        # 4.2 语言 Token Projection (直接截取后做维度对齐)
        self.vlm_lang_proj = nn.Linear(vlm_feat_dim, hidden_dim)
        
        # 4.3 DINO Adapter
        if self.use_dino:
            self.dino_adapter = VLMFeatureAdapter(
                vlm_feat_dim=dino_feat_dim,
                hidden_dim=hidden_dim,
                num_queries=vlm_num_queries,
                num_heads=num_heads,
                num_layers=adapter_depth,
                dropout=adapter_dropout_ratio
            )
        
        # 5. DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        
        # 6. Output Projection
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        
        # self.output_proj = nn.Linear(hidden_dim, action_dim)
        self.output_proj = nn.Conv1d(
            in_channels=hidden_dim, 
            out_channels=action_dim, 
            kernel_size=3, 
            padding=1
        )
        
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)

    def forward(self, t, noisy_actions, vlm_features, dino_features=None, qpos_history=None):
        """
        t: (B,)
        noisy_actions: (B, action_len, action_dim)
        vlm_features: (B, N, D_vlm)  N ~= 556
            Structure: [BOS (1) | Vision (512) | Language (Rest)]
        dino_features: (B, N_dino, D_dino) Optional
        qpos_history: (B, proprio_len, proprio_dim)
        """
        # --- 3. Time Embedding ---
        t_emb = self.time_mlp(t)
        
        
        # --- 1. 准备主序列 (Self-Attention Input) ---
        # 1.1 Action Embedding
        x_action = self.action_proj(noisy_actions) # (B, 32, H)
        x_action = x_action + self.action_pos_emb[:, :x_action.shape[1], :]

        # 1.2 Proprio Embedding
        x_proprio = self.proprio_proj(qpos_history) # (B, 8, H)
        x_proprio = x_proprio + self.proprio_pos_emb
        
        # 1.3 拼接 Action 和 Proprio 进入 Self-Attention
        # 这样 Action 可以在 self-attn 层直接关注到本体状态
        x = torch.cat([x_action, x_proprio], dim=1) # (B, 32+8, H)
        
        
        
        # 1.4 引入 Register Tokens
        # 将 Register Tokens 拼接到序列尾部，参与 Self-Attention 计算
        if self.num_registers > 0:
            B = x.shape[0]
            # (1, num_registers, H) -> (B, num_registers, H)
            registers = self.register_tokens.expand(B, -1, -1)
            # x shape: (B, action_len + proprio_len + num_registers, H)
            x = torch.cat([x, registers], dim=1)
        
        
        # TODO
        # 2.2 处理 Vision Token (Transformer Adapter 压缩)
        cond_vlm = self.vlm_visual_adapter(vlm_features)
     
        # 2.4 构建基础 Context
        c_full_list = [cond_vlm]
        
        # # 2.5 处理 DINO (如果启用)
        if self.use_dino and dino_features is not None:
            cond_dino = self.dino_adapter(dino_features)
            c_full_list.append(cond_dino)
        
        context_full = torch.cat(c_full_list, dim=1)
        
        # --- 4. DiT Forward ---
        for i, block in enumerate(self.blocks):
            # 根据当前层索引决定使用哪个 context
            if i in self.language_fusion_layers:
                current_context = context_full
            else:
                current_context = context_full
            
            x = block(x, t_emb, current_context)
            
        # --- 5. Output Projection ---
        x = self.final_norm(x)
        
        # 切分出 Action 部分进行预测
        x_action_out = x[:, :self.action_len, :] 
        
        # linear proj
        # pred_action = self.output_proj(x_action_out)
        
        
        # 准备 Conv1d 的输入格式: (B, C, L) -> (B, hidden_dim, action_len)
        x_action_out = x_action_out.transpose(1, 2)
        
        # 经过 1D 卷积输出: (B, action_dim, action_len)
        pred_action = self.output_proj(x_action_out)
    
        # 转置回目标格式: (B, action_len, action_dim)
        pred_action = pred_action.transpose(1, 2)
        
        return pred_action
    
    
    
    
    

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """标准的 DiT Block"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn1 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn2 = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, self.mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(self.mlp_hidden_dim, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, emb_t, cond_context):
        (shift_msa, scale_msa, gate_msa, 
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(emb_t).chunk(9, dim=1)

        # 1. Self-Attention (Input: Action + Proprio)
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn1(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # 2. Cross-Attention (Query: Action+Proprio, Key/Value: Vision+Lang+Dino)
        x_norm = modulate(self.norm2(x), shift_cross, scale_cross)
        attn_out, _ = self.attn2(x_norm, cond_context, cond_context)
        x = x + gate_cross.unsqueeze(1) * attn_out

        # 3. MLP
        x_norm = modulate(self.norm3(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(x_norm)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


def calc_flow_matching_loss(
    model, 
    x1, 
    vlm_features, 
    qpos_history, 
    qpos_target=None,
    dino_features=None,
    time_sampler="uniform",
    time_mu=0.0,      # Logit-Normal 的均值，0.0 对应 t=0.5 处的峰值
    time_sigma=1.0    # Logit-Normal 的标准差，越小越集中在 t=0.5
):
    """
    计算 Conditional Flow Matching Loss
    支持 uniform 和 logit_normal 时间步采样策略。
    """
    device = x1.device
    bs = x1.shape[0]
    
    # 1. 采样噪声 x0
    x0 = torch.randn_like(x1)
    
    # 2. 采样时间步 t
    if time_sampler == "uniform":
        t = torch.rand(bs, device=device)
    elif time_sampler == "logit_normal":
        # 采样标准正态分布
        normal_samples = torch.randn(bs, device=device)
        # 缩放和平移
        normal_samples = normal_samples * time_sigma + time_mu
        # 通过 sigmoid 映射到 (0, 1) 区间
        t = torch.sigmoid(normal_samples)
    else:
        raise ValueError(f"Unsupported time_sampler: {time_sampler}")
    
    # 3. 插值
    t_expand = t.view(bs, 1, 1)
    x_t = (1 - t_expand) * x0 + t_expand * x1
    
    # 4. 目标向量场
    target_v = x1 - x0 
    
    # 5. 模型预测
    pred_v = model(t, 
                   noisy_actions=x_t, 
                   vlm_features=vlm_features, 
                   dino_features=dino_features,
                   qpos_history=qpos_history
#                   qpos_target=qpos_target2026-07-07_19-38-43
                   )
    
    # 6. Loss
    loss = F.mse_loss(pred_v, target_v)
    
    return loss, {"pred_v": pred_v, "target_v": target_v}





