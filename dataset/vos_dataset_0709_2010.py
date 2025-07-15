import os
import json
import numpy as np
import cv2
import torch
import random
import logging

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from dataset.utils import sort_by_number, reseed

log = logging.getLogger(__name__)

class TenCamusDataset(Dataset):
    def __init__(self, filepath: str, mode: str = 'train', seq_length=15, max_num_obj=1, size=256, merge_probability=0.0):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        json_path = os.path.join(filepath, 'camus_public_datasplit_20250706.json')
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"{json_path} not found")

        with open(json_path, 'r') as f:
            all_data = json.load(f)
        if mode == 'train':
            self.patients = all_data['train_data']
        elif mode == 'val':
            self.patients = all_data['val_data']
        elif mode == 'test':
            self.patients = all_data['test_data']
        else:
            raise ValueError(f"Invalid mode: {mode}")

        # 收集所有样本（patient）信息
        self.samples = []
        for pid in self.patients:
            img_dir = os.path.join(self.filepath, 'img', pid)
            mask_dir = os.path.join(self.filepath, 'gt_lv', pid)

            if (not os.path.isdir(img_dir)) or (not os.path.isdir(mask_dir)):
                log.warning(f"{pid} 的图像目录或掩码目录不存在，已跳过。")
                continue

            img_list = sorted(os.listdir(img_dir), key=sort_by_number)
            mask_list = sorted(os.listdir(mask_dir), key=sort_by_number)

            if len(img_list) < self.seq_length:
                log.warning(f"{pid} 有 {len(img_list)} 帧图像，少于{self.seq_length}帧，已跳过。")
                continue

            self.samples.append({
                'patient_id': pid,
                'img_dir': img_dir,
                'mask_dir': mask_dir,
                'img_list': img_list,
                'mask_list': mask_list,
            })

        # 定义 sequence-level transform（对整段序列共享的变换）
        # 这里演示：先转成 PIL / Tensor 再做 torchvision 的增广
        if self.mode == 'train':
            self.seq_transform_img = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            # 测试时只缩放到 size
            self.seq_transform_img = transforms.Compose([
                # 这里可以只做 Resize 或不做任何操作
            ])

        # 定义 frame-level transform（每帧单独应用的）
        # 同样为了演示，这里只做 ToTensor() + Resize
        # 如果想跟 VOSMergeTrainDataset 一样，可以用 ColorJitter、RandomAffine 等
        self.frame_transform_img = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),  # => [0,1]
        ])

        # 注意：mask 不需要 ToTensor() 做 /255，因为它是标签，保持 0/1 即可
        # 但为了pipeline一致，可以先转成 PIL，再做 Resize，最后转成 long tensor
        self.frame_transform_mask = transforms.Compose([
            transforms.Resize((self.size, self.size), interpolation=transforms.InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        patient_id = sample['patient_id']
        img_dir = sample['img_dir']
        mask_dir = sample['mask_dir']
        img_list = sample['img_list']

        T = self.seq_length
        total_frames = len(img_list)

        if total_frames > T:
            start_frame_idx = random.randint(0, total_frames - T)
        else:
            start_frame_idx = 0
        frame_indices = range(start_frame_idx, start_frame_idx + T)

        frames = []
        masks = []

        for current_frame_idx in frame_indices:
            img_name = img_list[current_frame_idx]
            img_path = os.path.join(img_dir, img_name)
            
            # --- START of CHANGE: 为每一帧加载掩码 ---
            # 移除了只加载首尾帧掩码的限制。
            # 现在会尝试为序列中的每一帧加载同名掩码文件。
            mask_img = np.zeros((256, 256), dtype=np.uint8) # 默认使用全黑的掩码
            
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.isfile(mask_path):
                loaded_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if loaded_mask is not None:
                    mask_img = loaded_mask
                else:
                    log.warning(f"Failed to read mask from {mask_path}, using zeros.")
            # 如果掩码文件不存在，则静默使用全零掩码。
            # --- END of CHANGE ---

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                log.warning(f"Failed to read image from {img_path}, using zeros.")
                image = np.zeros((256, 256), dtype=np.uint8)

            # 转成 PIL，后面用 torchvision transforms
            pil_img = to_pil_image(image)
            pil_mask = to_pil_image(mask_img)

            frames.append(pil_img)
            masks.append(pil_mask)

        # ----------- 在这里做 sequence-level transform -----------
        # 例如：对整个序列使用同一个随机变换
        # 做法是：先采一个随机种子，然后对每帧都用这同一个变换
        seq_seed = random.randint(0, 99999)
        transformed_frames = []
        transformed_masks = []

        for i in range(T):
            # 对图像
            reseed(seq_seed)
            f_img = self.seq_transform_img(frames[i])
            transformed_frames.append(f_img)

            # 对掩码需要相同的随机操作（如果在 torchvision 里，需要自己保证翻转/旋转与图像一致）
            # 但 torchvision 的 transforms.RandomHorizontalFlip() / RandomRotation() 对 mask 不同步处理并不方便
            # 你可以换成 Albumentations 并设置相同random_seed
            # 这里演示简单情况：同一个 transforms 也应用到 mask
            reseed(seq_seed)
            f_mask = self.seq_transform_img(masks[i])
            transformed_masks.append(f_mask)

        # ----------- 再做 frame-level transform -----------
        # 每帧独立应用，如 ToTensor()、Resize
        final_imgs = []
        final_masks = []
        for i in range(T):
            # 图像
            img_t = self.frame_transform_img(transformed_frames[i])  # => [C,H,W], float in [0,1]
            # mask
            mask_pil = self.frame_transform_mask(transformed_masks[i])
            mask_np = np.array(mask_pil, dtype=np.uint8)
            mask_np[mask_np != 1] = 0  # 如果只有前景=1，其它=0
            # mask_np[mask_np > 0] = 1  # 如果有多个前景，可以这样处理
            mask_t = torch.from_numpy(mask_np).long().unsqueeze(0)  # => shape [1,H,W]

            final_imgs.append(img_t)
            final_masks.append(mask_t)

        # 拼成 [T, C, H, W] / [T, 1, H, W]
        final_imgs_t = torch.stack(final_imgs, dim=0)   # => [T, C, H, W]
        final_masks_t = torch.stack(final_masks, dim=0) # => [T, 1, H, W]

        # ----------- 构造和 VOS 相似的输出结构 ------------
        # info 里面放一些元信息
        info = {
            'name': patient_id,
            'frames': [img_list[i] for i in range(T)]
        }

        # 1) 从第一帧掩码中找出所有非 0 标签 => target_objects
        #    final_masks_t[0] 形状 [1,H,W]
        first_frame_labels = final_masks_t[0].unique()
        first_frame_labels = [v.item() for v in first_frame_labels if v.item() != 0]
        # 截断到 max_num_obj
        target_objects = first_frame_labels[:self.max_num_obj]

        # 2) 构造 cls_gt: 与 final_masks_t 同形状 [T,1,H,W], 先置0
        cls_gt = torch.zeros_like(final_masks_t)  # dtype long, shape [T,1,H,W]

        # 3) 构造 first_frame_gt: [1, max_num_obj, H, W]
        first_frame_gt = torch.zeros(
            (1, self.max_num_obj, self.size, self.size), dtype=torch.long
        )

        # 4) 遍历每一个目标 ID，进行赋值
        #    - cls_gt[this_mask] = i+1
        #    - first_frame_gt[0,i] = this_mask[0,0]
        for i, l in enumerate(target_objects):
            # this_mask => 布尔张量 [T,1,H,W]
            this_mask = (final_masks_t == l)

            # cls_gt 中，把属于这个对象的像素置为 i+1
            cls_gt[this_mask] = i + 1

            # 第一帧第 i 个通道是这个对象的二值掩码
            # this_mask[0] => [1,H,W] => this_mask[0,0] => [H,W]
            first_frame_gt[0, i] = this_mask[0, 0].long()

        # 5) selector: [max_num_obj], 对实际出现的目标设 1，其余 0
        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)
        selector[:len(target_objects)] = 1.0

        # num_objects (仅做记录, 可有可无)
        info['num_objects'] = len(target_objects)

        data = {
            'rgb'      : final_imgs_t,    # [T,C,H,W], float in [0,1]
            'ff_gt'    : first_frame_gt,  # [T=1, num_objects, H, W]
            'cls_gt'   : cls_gt,          # [T,1,H,W], long
            'selector' : selector,        # [1]
            'info'     : info,
        }

        return data


