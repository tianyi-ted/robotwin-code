import torch
import logging

# 初始化 logger
logger = logging.getLogger(__name__)



# =================================================================================
# 新增: 训练前 Batch 预览与统计 (包含维度、Min/Max)
# =================================================================================
def inspect_batch_stats(dataloader, logger):
    logger.info("Fetching a single batch for inspection...")
    try:
        batch = next(iter(dataloader))
        
        print("\n" + "="*40 + " Batch Inspection (With Min/Max) " + "="*40)
        
        # 辅助函数：打印单个 Tensor 的信息
        def print_tensor_info(k, v, indent=0):
            prefix = " " * indent
            if isinstance(v, torch.Tensor):
                # 对于浮点数或整数计算 Min/Max
                if torch.is_floating_point(v) or torch.is_complex(v) or v.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32, torch.float64, torch.bfloat16]:
                    min_val = f"{v.min().item():.4f}"
                    max_val = f"{v.max().item():.4f}"
                else:
                    min_val = "N/A"
                    max_val = "N/A"
                
                shape_str = str(list(v.shape))
                print(f"{prefix}Key: {k:<20} | Shape: {shape_str:<22} | Min: {min_val:<10} | Max: {max_val:<10} | Type: {v.dtype}")
            
            elif v is None:
                print(f"{prefix}Key: {k:<20} | Type: NoneType")
            
            elif isinstance(v, list):
                print(f"{prefix}Key: {k:<20} | Type: List (len={len(v)})")
            
            else:
                print(f"{prefix}Key: {k:<20} | Type: {type(v)}")

        # 1. 打印顶层键值对
        keys = sorted(batch.keys())
        for key in keys:
            value = batch[key]
            if isinstance(value, dict):
                print(f"Key: {key:<20} | Type: Dict (Nested)")
            else:
                print_tensor_info(key, value)

        # 2. 专门处理 vlm_inputs (嵌套字典)
        if 'vlm_inputs' in batch and isinstance(batch['vlm_inputs'], dict):
            print("-" * 118)
            print("Nested Key: ['vlm_inputs']")
            vlm_batch = batch['vlm_inputs']
            for sub_key in sorted(vlm_batch.keys()):
                print_tensor_info(sub_key, vlm_batch[sub_key], indent=4)
        
        print("="*118 + "\n")

    except Exception as e:
        logger.error(f"Failed to inspect batch: {e}")



def count_parameters(model, model_name="Model"):
    """
    计算并打印模型参数量 (单位: M)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    total_m = total_params / 1e6
    trainable_m = trainable_params / 1e6
    
    logger.info(f"[{model_name}] Stats:")
    logger.info(f"  - Total Parameters: {total_m:.2f}M")
    logger.info(f"  - Trainable Parameters: {trainable_m:.2f}M")
    
    return total_m, trainable_m

