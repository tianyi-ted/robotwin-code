# Robotwin2 Dataset Loader for Motus (HDF5 Version)
# Supports Robotwin2 data with HDF5 storage and flexible indexing
import os
import random
import h5py
import numpy as np
import cv2
import json
from tqdm import tqdm
import torch
import torch.utils.data as data
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from pathlib import Path
import warnings
import torchvision.transforms as T
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# VLM processing imports
from utils.vlm_utils import preprocess_vlm_messages
from transformers import AutoProcessor

warnings.filterwarnings("ignore", category=FutureWarning, message=".*multichannel.*")

logger = logging.getLogger(__name__)

# 定义线程数，用于图像解码
NUM_THREADS = os.cpu_count() or 4

def _decode(buf):
    """
    Helper function for parallel image decoding.
    Decodes bytes to RGB numpy array.
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    # cv2.imdecode returns BGR
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)   
    # Convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class RobotWinTaskDataset(data.Dataset):
    def __init__(self, dataset_dir, data_mode="clean", 
                 indices_config=None, camera_names=None, video_size=(320, 240), 
                 val=False, image_aug=False, vlm_checkpoint_path=None, stage1_mode=False):
        
        self.dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, str) else [Path(p) for p in dataset_dir]
        self.data_mode = data_mode
        self.stage1_mode = stage1_mode
        self.all_episodes = [] # 用于存储所有扫描到的 episode 信息
        
        if indices_config is None: raise ValueError("indices_config is required")
        if 'state_indices' not in indices_config or 'action_indices' not in indices_config:
            raise ValueError("indices_config missing required keys")

        self.state_offsets = torch.tensor(indices_config['state_indices'], dtype=torch.long)
        self.action_offsets = torch.tensor(indices_config['action_indices'], dtype=torch.long)
        self.chunk_size = len(self.action_offsets)
        self.indices_config = indices_config
        self.camera_names = camera_names or ['head_camera']
        self.video_size = video_size 
        self.val = val
        self.image_aug = image_aug
        
        
        
        # 使用 ColorJitter 随机改变亮度、对比度、饱和度、色调，不改变语义
        self.aug_pool = []
        if self.image_aug and not self.val:
            logger.info("Initializing Image Augmentation (Randomly picking 1-2 ops)...")
            self.aug_pool = [
                T.ColorJitter(brightness=0.05),
                T.ColorJitter(contrast=0.05),
                T.ColorJitter(saturation=0.05),
                T.ColorJitter(hue=0.05)
            ]
        
        
        # 如果是 Stage 1，不需要初始化 VLM Processor
        if stage1_mode:
            self.vlm_processor = None
        else:
            self.vlm_processor = AutoProcessor.from_pretrained(
                vlm_checkpoint_path,
                local_files_only=True
            )


        # 1. 扫描文件夹，加载所有文件的基本路径信息
        self._load_episodes()

        # 2. 无论 Stage 1 还是 Stage 2，都统一建立平铺索引 (Flattened Index Map)
        logger.info("Building Index Map for dataset...")
        self._build_index_map()
        
        # 3. Stage 1 特有：预加载低维数据到 RAM
        if self.stage1_mode:
            logger.info("Stage 1: Preloading all State and Action data into RAM...")
            self._preload_stage1_data()

    # --- 统一的数据索引构建 ---
    def _build_index_map(self):
        """
        遍历所有 episode，收集元数据并构建平铺索引。
        将所有帧的绝对索引存入 valid_indices，确保长轨迹被采样的概率更高。
        """
        self.valid_indices = [] # 存储全局绝对索引列表
        self.episode_metadata = [] # 存储 (filepath, length, instruction_path)等

        current_offset = 0
        valid_ep_count = 0

        for ep_info in tqdm(self.all_episodes, desc="Scanning episode lengths"):
            path = ep_info['hdf5_path']
            
            with h5py.File(path, 'r') as f:
                length = f['joint_action']['vector'].shape[0]
                
            if length < 2: continue # 跳过过短的数据

            self.episode_metadata.append({
                'hdf5_path': path,
                'json_path': ep_info['json_path'],
                'length': length,
                'global_start': current_offset,
                'global_end': current_offset + length
            })
            
            ep_start = current_offset
            ep_end = current_offset + length
            
            # 将该 episode 内的所有帧放入全局有效索引池
            curr_indices = np.arange(ep_start, ep_end, dtype=np.int64)
            self.valid_indices.append(curr_indices)
            
            current_offset += length
            valid_ep_count += 1
                
 
                
        self.valid_indices = np.concatenate(self.valid_indices)
        
        # 建立快速查找表: 全局索引 -> episode_idx
        # 利用 searchsorted 可以 O(log N) 找到某帧属于哪个 episode
        self._ep_end_bounds = np.array([ep['global_end'] for ep in self.episode_metadata])
        
        logger.info(f"Index map built: {valid_ep_count} valid episodes, {len(self.valid_indices)} total searchable frames.")

    def _preload_stage1_data(self):
        """Stage 1: 多线程读取所有 Action 和 State 到内存"""
        temp_actions, temp_states = [], []
        
        def load_single_file(ep_meta):
            try:
                with h5py.File(ep_meta['hdf5_path'], 'r') as f:
                    action = f['joint_action']['vector'][:].astype(np.float32)
                    l_pose = f['endpose']['left_endpose'][:]
                    l_grip = f['endpose']['left_gripper'][:]
                    r_pose = f['endpose']['right_endpose'][:]
                    r_grip = f['endpose']['right_gripper'][:]
                    if l_grip.ndim == 1: l_grip = l_grip[:, None]
                    if r_grip.ndim == 1: r_grip = r_grip[:, None]
                    state = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1).astype(np.float32)
                    return action, state
            except Exception: return None

        with ThreadPoolExecutor(max_workers=NUM_THREADS*2) as executor:
            futures = [executor.submit(load_single_file, meta) for meta in self.episode_metadata]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preloading Stage 1 RAM"):
                res = future.result()
                if res is not None:
                    act, sta = res
                    temp_actions.append(act)
                    temp_states.append(sta)

        self.buffer_action = torch.from_numpy(np.concatenate(temp_actions, axis=0))
        self.buffer_state = torch.from_numpy(np.concatenate(temp_states, axis=0))

    # --- 文件扫描 ---
    def _scan_task_folder(self, task_path: Path, split_name: str) -> List[Dict[str, Any]]:
        data_dir = task_path / "data"
        instruct_dir = task_path / "instructions"
        if not data_dir.exists(): return []
        valid_episodes = []
        for hdf5_path in data_dir.glob("*.hdf5"):
            json_path = instruct_dir / f"{hdf5_path.stem}.json"
            if json_path.exists():
                valid_episodes.append({
                    'episode_name': hdf5_path.stem,
                    'task_name': task_path.parent.name if task_path.name in ['demo_clean', 'demo_randomized'] else task_path.name,
                    'hdf5_path': str(hdf5_path), 'json_path': str(json_path), 'split': split_name
                })
        return valid_episodes

    def _load_episodes(self):
        """遍历指定的根目录，查找所有包含数据的 Task 文件夹，并将所有 episode 压入列表"""
        logger.info("Scanning dataset folders for all tasks...")
        data_splits = ["demo_clean", "demo_randomized"] if self.data_mode == "both" else [f"demo_{self.data_mode}"] 
        
        for root_dir in self.dataset_dirs:
            if not root_dir.exists(): continue
            # 遍历每一个任务文件夹
            for task_dir in [d for d in root_dir.iterdir() if d.is_dir()]:
                for split in data_splits:
                    split_path = task_dir / split
                    if split_path.exists():
                        episodes = self._scan_task_folder(split_path, split)
                        self.all_episodes.extend(episodes)

        if not self.all_episodes: 
            raise ValueError(f"No valid episodes found in any of the provided dataset directories: {self.dataset_dirs}")
        
        logger.info(f"Successfully scanned {len(self.all_episodes)} total episode files.")

    # --- 辅助方法 ---
    def _load_text_instruction(self, json_path: str) -> str:
        try:
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            instructions = data.get('seen', [])
            if not instructions: raise ValueError("No instructions found")
            return random.choice(instructions)
        except Exception: return "do something" 

    def _get_query_indices(self, query_idx: int, episode_len: int) -> Tuple[Dict[str, List[int]], Dict[str, torch.Tensor]]:
        ep_start, ep_end = 0, episode_len
        query_indices, padding_mask = {}, {}
        keys_to_process = {'state': self.indices_config['state_indices'], 'action': self.indices_config['action_indices']}
        for cam_name in self.camera_names: keys_to_process[cam_name] = self.indices_config['camera_indices']

        for key, delta_list in keys_to_process.items():
            abs_indices = [query_idx + delta for delta in delta_list]
            query_indices[key] = [max(ep_start, min(ep_end - 1, idx)) for idx in abs_indices]
            
            if key == 'action':
                valid_mask = [(idx < ep_end) for idx in abs_indices]
                padding_mask[f"{key}_mask"] = torch.from_numpy(np.array(valid_mask, dtype=bool))
            else:
                padding_mask[f"{key}_mask"] = torch.ones(len(abs_indices), dtype=torch.bool)
                
        return query_indices, padding_mask

    def _load_hdf5_data(self, hdf5_path: str, query_indices: Dict[str, List[int]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        with h5py.File(hdf5_path, 'r') as root:
            # Actions
            t_idx = np.array(query_indices['action'])
            h5_idx = np.unique(t_idx)
            data_batch['action_sequence'] = torch.from_numpy(root['joint_action']['vector'][h5_idx][np.searchsorted(h5_idx, t_idx)]).float()

            # State
            t_idx = np.array(query_indices['state'])
            h5_idx = np.unique(t_idx)
            l_pose = root['endpose']['left_endpose'][h5_idx]
            l_grip = root['endpose']['left_gripper'][h5_idx]
            r_pose = root['endpose']['right_endpose'][h5_idx]
            r_grip = root['endpose']['right_gripper'][h5_idx]
            if l_grip.ndim == 1: l_grip = l_grip[:, None]
            if r_grip.ndim == 1: r_grip = r_grip[:, None]
            state_data = np.concatenate([l_pose, l_grip, r_pose, r_grip], axis=1)
            data_batch['state'] = torch.from_numpy(state_data[np.searchsorted(h5_idx, t_idx)]).float()

            # Cameras
            data_batch['frame'] = {}
            for cam in self.camera_names:
                t_idx = np.array(query_indices[cam])
                if cam not in root['observation']: continue
                h5_idx = np.unique(t_idx)
                comp_imgs = root['observation'][cam]['rgb'][h5_idx]
                
                # with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
                    # decoded = np.stack(list(pool.map(_decode, comp_imgs)))
                decoded = np.stack([_decode(img) for img in comp_imgs])
                
                final_img = np.ascontiguousarray(decoded[np.searchsorted(h5_idx, t_idx)])
                if (final_img.shape[2], final_img.shape[1]) != self.video_size:
                    final_img = np.stack([cv2.resize(i, self.video_size, interpolation=cv2.INTER_LINEAR) for i in final_img])
                
                
                # # To Tensor [T, C, H, W] and Normalize
                # img_tensor = torch.from_numpy(final_img).permute(0, 3, 1, 2).float() / 255.0
                
                # # 应用图像增强
                # if self.color_jitter is not None:
                #     img_tensor = self.color_jitter(img_tensor)
                
                
                data_batch['frame'][cam] = final_img
        return data_batch

    # --- Pytorch Dataloader Interfaces ---
    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        # O(1) 获取全局索引
        global_curr_idx = self.valid_indices[idx]
        # import time
        # ss = time.time()
        # 二分查找确定属于哪一个 episode  1.5e-5 second
        ep_idx = np.searchsorted(self._ep_end_bounds, global_curr_idx, side='right')
        ep_meta = self.episode_metadata[ep_idx]
        
        abs_start = ep_meta['global_start']
        abs_end = ep_meta['global_end']
        
        # ================= Stage 1 高速路径 (内存截取) =================
        if self.stage1_mode:
            t_state_idx = global_curr_idx + self.state_offsets
            clamp_state_idx = torch.clamp(t_state_idx, min=abs_start, max=abs_end - 1)
            state_seq = self.buffer_state[clamp_state_idx]
            state_mask = torch.ones(len(self.state_offsets), dtype=torch.bool)

            t_action_idx = global_curr_idx + self.action_offsets
            action_mask = t_action_idx < abs_end 
            read_indices = torch.clamp(t_action_idx, max=abs_end - 1)
            action_seq = self.buffer_action[read_indices]

            return {
                'action_sequence': action_seq, 'state': state_seq,
                'action_mask': action_mask, 'state_mask': state_mask,
                'vlm_inputs': None
            }

        # ================= Stage 2 完整路径 (磁盘读取) =================
        try:
            # 换算为单个 episode 内的局部索引
            local_anchor_idx = global_curr_idx - abs_start
            total_frames = ep_meta['length']
            
            # 计算局部查询索引
            query_indices, padding_mask = self._get_query_indices(local_anchor_idx, total_frames)
            
            # 加载 HDF5 和 图像
            data_batch = self._load_hdf5_data(ep_meta['hdf5_path'], query_indices)
            data_batch.update(padding_mask)
            
            # 文本与 VLM
            instruction = self._load_text_instruction(ep_meta['json_path'])
            
            primary_cam = self.camera_names[0]
            if primary_cam in data_batch['frame']:
                
                # --- [新增] 随机选择 1 到 2 个增强操作 ---
                # 放在 for 循环外面，保证同一序列里的几帧图像经历的增强“种类”是一致的
                active_ops = []
                if self.aug_pool:
                    num_ops = random.choice([1, 2]) # 决定抽1个还是2个
                    active_ops = random.sample(self.aug_pool, num_ops) # 不放回抽样
                
                # [优化核心] 将 Numpy 直接转为 PIL 并进行增强
                pil_imgs = []
                for img_np in data_batch['frame'][primary_cam]:
                    pil_img = Image.fromarray(img_np)
                    
                    # --- 依次应用被选中的 1~2 个操作 ---
                    for op in active_ops:
                        pil_img = op(pil_img)
                    
                    pil_imgs.append(pil_img)
                    
                vlm_inputs = preprocess_vlm_messages(instruction, pil_imgs, self.vlm_processor)
        
            # 组装返回 (注意，由于我们将 frame 去掉了 tensor 转换，目前 Dataloader 返回的 result 中不再包含原始 frame)
            # 因为只有 VLM 需要图像，我们没有必要保留未经 Processor 处理的 frame 副本。如果确有其它需求，也可以保留。
            result = {
                'state': data_batch['state'],
                'action_sequence': data_batch['action_sequence'],
                'vlm_inputs': vlm_inputs,
                'state_mask': data_batch['state_mask'],
                'action_mask': data_batch['action_mask'],
            }
                
            return result
            
        except Exception as e:
            # 错误恢复：遇到损坏的文件随机重试
            logger.warning(f"Error loading idx {idx}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor [C, H, W] to PIL Image."""
    if tensor.shape[0] == 3:
        image_np = tensor.permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)
        return Image.fromarray(image_np, mode='RGB')
    raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

# Factory Function
def create_dataset(config: Any, val: bool = False, stage1_mode: bool = False):
    # Extract indices config
    indices_config = None
    if hasattr(config.dataset, 'indices_config'):
        from omegaconf import OmegaConf
        indices_config = OmegaConf.to_container(config.dataset.indices_config, resolve=True)

    params = {
        'dataset_dir': config.dataset.dataset_dir, 
        'indices_config': indices_config,
        'val': val,
        'stage1_mode': stage1_mode,
        'image_aug': config.dataset.get('image_aug', False) and not val,
        'camera_names': config.dataset.get('camera_names', ['head_camera']) # Default to head_camera
    }
    
    for key in ['data_mode']:
        if hasattr(config.dataset, key):
            params[key] = getattr(config.dataset, key)
            
    # 仅在Stage 2 模式下才传入 VLM 路径
    # 这样 Stage 1 时 Dataset 就不会加载 VLM Processor，节省内存和启动时间
    if not stage1_mode:
        params['vlm_checkpoint_path'] = config.model.vlm.checkpoint_path

    return RobotWinTaskDataset(**params)


def _process_vlm_inputs_batch(vlm_inputs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Process and batch VLM inputs with padding."""
    input_ids_list = [x['input_ids'] for x in vlm_inputs]
    pixel_values_list = [x.get('pixel_values') for x in vlm_inputs]
    image_grid_thw_list = [x.get('image_grid_thw') for x in vlm_inputs]
    attention_mask_list = [x.get('attention_mask') for x in vlm_inputs]
    mm_token_type_ids_list = [x.get('mm_token_type_ids') for x in vlm_inputs]
    
    max_seq_len = max(ids.shape[1] for ids in input_ids_list)
    padded_input_ids = []
    padded_attention_masks = []
    padded_mm_token_type_ids = []
    
    for i, ids in enumerate(input_ids_list):
        mask = attention_mask_list[i]
        mm_tids = mm_token_type_ids_list[i]
        
        if ids.shape[1] < max_seq_len:
            padding_size = max_seq_len - ids.shape[1]
            padding = torch.zeros(ids.shape[0], padding_size, dtype=ids.dtype, device=ids.device)
            padded_ids = torch.cat([ids, padding], dim=1)
            
            if mask is not None:
                mask_padding = torch.zeros(mask.shape[0], padding_size, dtype=mask.dtype, device=mask.device)
                padded_mask = torch.cat([mask, mask_padding], dim=1)
            else:
                padded_mask = None

            if mm_tids is not None:
                mm_padding = torch.zeros(mm_tids.shape[0], padding_size, dtype=mm_tids.dtype, device=mm_tids.device)
                padded_mm_tids = torch.cat([mm_tids, mm_padding], dim=1)
            else:
                padded_mm_tids = None
        else:
            padded_ids = ids
            padded_mask = mask
            padded_mm_tids = mm_tids
            
        padded_input_ids.append(padded_ids)
        padded_attention_masks.append(padded_mask)
        padded_mm_token_type_ids.append(padded_mm_tids)
    
    return {
        'input_ids': torch.cat(padded_input_ids, dim=0),
        # 仅当 pixel_values 存在且不为 None 时才拼接
        'pixel_values': torch.cat([pv for pv in pixel_values_list if pv is not None], dim=0) if pixel_values_list and any(pv is not None for pv in pixel_values_list) else None,
        'image_grid_thw': torch.cat([igt for igt in image_grid_thw_list if igt is not None], dim=0) if image_grid_thw_list and any(igt is not None for igt in image_grid_thw_list) else None,
        'attention_mask': torch.cat([mask for mask in padded_attention_masks if mask is not None], dim=0) if any(mask is not None for mask in padded_attention_masks) else None,
        'mm_token_type_ids': torch.cat([mm for mm in padded_mm_token_type_ids if mm is not None], dim=0) if any(mm is not None for mm in padded_mm_token_type_ids) else None,
    }


def collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0: return None
    
    result = {}
    # 获取第一个样本的所有 key (包含 frame_xxx, xxx_mask, state, action 等)
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'vlm_inputs':
            # VLM 数据特殊处理
            vlm_inputs = [sample.get('vlm_inputs') for sample in batch]
            if vlm_inputs and all(x is not None for x in vlm_inputs):
                result['vlm_inputs'] = _process_vlm_inputs_batch(vlm_inputs)
            else:
                result['vlm_inputs'] = None
        else:
            # 对于其他所有 Tensor 数据（动态相机名、Mask、State、Action），直接 Stack
            # 假设这些 key 对应的值都是 Tensor
            val = batch[0][key]
            if isinstance(val, torch.Tensor):
                result[key] = torch.stack([sample[key] for sample in batch])
            elif val is None:
                result[key] = None
    
    return result







