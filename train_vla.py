import os
import sys
import torch
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- 引入项目依赖 ---
# 假设 dataset.py 在 hdf5_dataloader 文件夹下
from hdf5_dataloader.dataset import RobotWinTaskDataset, collate_fn, create_dataset
# 假设 model_runner.py 包含 ModelFactory 和 VLAWrapperTwoStage
from models.model_runner import ModelFactory, VLAWrapperTwoStage
from utils.train_utils import inspect_batch_stats, count_parameters

# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载并处理配置"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    return config


if __name__ == "__main__":
    # =========================================================================
    # 1. 参数解析 (Arguments)
    # =========================================================================
    parser = argparse.ArgumentParser(description="VLA Two-Stage Training Script")
    
    # 基础配置
    parser.add_argument("--config", type=str, default="./configs/robotwin.yaml", help="Path to config file")
    parser.add_argument("--norm_stats_path", type=str, default="./utils/stat-200-10.json", help="Path to normalization stats")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_vla", help="Directory to save checkpoints")
    
    # 训练超参
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save_interval", type=int, default=1, help="Save checkpoint every N epochs")

    # 两阶段训练相关参数
    parser.add_argument("--stage1", default=False, help="Enable Stage 1 (Motor Babbling) mode")
    
    # for state 2 training 
    parser.add_argument("--stage1_checkpoint", type=str, 
                        default='./checkpoints_vla/stage1/2026-02-22_01-45-54/checkpoint_epoch_25.pt', 
                        help="Path to Stage 1 checkpoint (Required for Stage 2)"
                        )
    parser.add_argument("--freeze_stage1", default=False, help=" Stage 1 weights will be trainable in Stage 2")
    
    args = parser.parse_args()

    # =========================================================================
    # 2. 环境设置
    # =========================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 推荐使用 bfloat16 进行训练，特别是 DiT 和 VLM
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    logger.info(f"Using device: {device}, Precision: {dtype}")
    
    config = load_config(args.config)

    # =========================================================================
    # 3. 数据集准备 (Dataset & Dataloader)
    # =========================================================================
    
    if args.stage1:
        batch_size = config.training.batch_size_stage1
    else:
        batch_size = config.training.batch_size

    logger.info(f"Creating Dataset (Stage 1 Mode: {args.stage1})...")
    
    # 核心：传递 stage1_mode=True 会触发内存预加载和高速索引
    train_dataset = create_dataset(config, val=False, stage1_mode=args.stage1)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # Stage 1 数据在内存中，num_workers 不需要太多，甚至 0 都可以，但设为 4-8 可以预取
        num_workers=config.system.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    logger.info(f"Dataset Size: {len(train_dataset)} samples/episodes")
    logger.info(f"Batches per Epoch: {len(train_dataloader)}")

    # =========================================================================
    # 4. 模型初始化 (Model Factory)
    # =========================================================================
    if args.stage1:
        logger.info(">>> Initializing STAGE 1: Motor Babbling (No VLM)")
        
        # Stage 1 不需要 VLM
        vlm_model = None
        vlm_hidden_size = 0 # 占位符，Stage 1 模型不使用它
        
        # 创建 Stage 1 Action Model
        action_model = ModelFactory.create_action_model_two_stage(
            config, 
            vlm_hidden_size=None,
            stage=1,
            stage1_ckpt_path=None,
        )
        
    else:
        logger.info(">>> Initializing STAGE 2: Instruction Following (With VLM)")
        
        # Stage 2 必须加载 VLM
        vlm_model = ModelFactory.create_vlm(config.model.vlm.checkpoint_path, dtype, device)
        vlm_hidden_size = vlm_model.config.text_config.hidden_size
        
        # 创建 Stage 2 Action Model
        # 如果提供了 checkpoint，Factory 会自动加载权重并冻结 Stage 1 参数
        if args.stage1_checkpoint is None:
            logger.warning("WARNING: running Stage 2 without loading Stage 1 checkpoint!")
            
        action_model = ModelFactory.create_action_model_two_stage(
            config, 
            vlm_hidden_size=vlm_hidden_size,
            stage=2,
            stage1_ckpt_path=args.stage1_checkpoint,
            freeze_stage1=args.freeze_stage1
        )

    # 将 Action Model 移动到 GPU
    action_model.to(device, dtype=dtype)
    action_model.train()
    
    # 打印参数量
    count_parameters(action_model, model_name="Action Model (Trainable)")
    
    model = VLAWrapperTwoStage(
        vlm_model=vlm_model,
        action_model=action_model,
        time_sampler=config.training.time_sampler,
        visual_feat_layer=config.model.visual_feat_layer,
        vlm_feat_layer=config.model.vlm_feat_layer,
        device=device,
        dtype=dtype,
        norm_stats_path=args.norm_stats_path
    )

    # =========================================================================
    # 6. 优化器 (Optimizer)
    # =========================================================================
    
    # 定义两个参数列表
    params_stage1 = [] # 存放 Self-Attn, MLP, Embeddings
    params_stage2 = [] # 存放 Cross-Attn, Adapter, Projector

    # 遍历 Action Model 的参数
    # 注意：action_model 是 VLAStage1 或 VLAStage2 实例
    for name, param in action_model.named_parameters():
        if not param.requires_grad:
            continue # 跳过冻结参数
        
        # 如何区分 Stage 1 和 Stage 2？
        # 在 VLAStage2 中，Stage 1 的模型被存储在 self.stage1 属性下
        # 因此，所有 Stage 1 参数的名字都以 "stage1." 开头
        if name.startswith("stage1."):
            params_stage1.append(param)
        else:
            # 其他参数（如 vlm_proj, blocks.x.attn_cross, visual_adapter 等）属于 Stage 2
            params_stage2.append(param)

    # 打印分组信息
    logger.info(f"Optimizer Grouping:")
    logger.info(f"  - Group Stage 2 (New params): {len(params_stage2)} tensors, LR = {config.training.learning_rate}")
    if len(params_stage1) > 0:
        logger.info(f"  - Group Stage 1 (Finetune): {len(params_stage1)} tensors, LR = {config.training.learning_rate_finetune_stage1}")
    else:
        logger.info(f"  - Group Stage 1 (Frozen): 0 tensors")

    # 构建参数组
    optim_groups = [
        # Group 0: Stage 2 参数 (正常学习率)
        {'params': params_stage2, 'lr': config.training.learning_rate},
    ]
    
    # Group 1: Stage 1 参数 (微调学习率)，仅当参数存在时添加
    if len(params_stage1) > 0:
        optim_groups.append({'params': params_stage1, 'lr': config.training.learning_rate_finetune_stage1})

    # 初始化优化器
    optimizer = AdamW(
        optim_groups,
        betas=config.training.betas,
        weight_decay=config.training.weight_decay
    )
    
    parameters_to_clip = []
    for group in optimizer.param_groups:
        parameters_to_clip.extend(group['params'])
    
    # Scheduler 需要知道总步数
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_dataloader), 
        eta_min=1e-6
    )

    # =========================================================================
    # 7. 训练循环 (Training Loop)
    # =========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stage_name = "stage1" if args.stage1 else "stage2"
    run_save_dir = os.path.join(args.save_dir, stage_name, timestamp)
    os.makedirs(run_save_dir, exist_ok=True)
    
    # 保存当前的 Config 备份
    OmegaConf.save(config, os.path.join(run_save_dir, "config.yaml"))
    logger.info(f"Checkpoints will be saved to: {run_save_dir}")

    global_step = 0

    for epoch in range(args.epochs):
        model.train() 
        epoch_loss = 0.0
        optimizer.zero_grad()
        
        start_time = time.time()

        for step, batch in enumerate(train_dataloader):
            # 处理部分无效 batch (虽然 collate_fn 已经过滤了)
            if batch is None: continue
            
            # 使用 autocast 自动处理精度
            with torch.amp.autocast('cuda', dtype=dtype):
                # Wrapper 会根据内部状态决定是否提取 VLM 特征
                # Stage 1: Action/State -> Norm -> Loss
                # Stage 2: VLM -> Features -> Action/State -> Norm -> Loss
                loss = model(batch)
                
                loss = loss / args.grad_accum_steps
            
            # Backward
            loss.backward()
            epoch_loss += loss.item() * args.grad_accum_steps
            
            # Gradient Accumulation Step
            if (step + 1) % args.grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % 20 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Epoch [{epoch+1}/{args.epochs}] "
                        f"Step [{step+1}/{len(train_dataloader)}] "
                        f"Loss: {loss.item() * args.grad_accum_steps:.4f} "
                        f"LR: {current_lr:.2e} "
                    )

        # End of Epoch
        avg_loss = epoch_loss / len(train_dataloader)
        logger.info(f"=== Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f} ===")

        # Checkpoint Saving
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            ckpt_name = f"checkpoint_epoch_{epoch+1}.pt"
            ckpt_path = os.path.join(run_save_dir, ckpt_name)
            
            save_dict = {
                'epoch': epoch + 1,
                # 注意：我们只保存 Action Model 的参数
                # 在 Stage 2，这也是包含 Cross Attn 的完整模型，但不包含 VLM
                'model_state_dict': action_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'stage': 1 if args.stage1 else 2
            }
            
            torch.save(save_dict, ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    logger.info("Training Complete.")