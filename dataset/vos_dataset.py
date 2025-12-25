import os
import json
import logging
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from dataset.utils import sort_by_number

log = logging.getLogger(__name__)

class TenCamusDataset(Dataset):
    def __init__(self, filepath: str, mode: str = 'train', seq_length=10, max_num_obj=1, size=256, merge_probability=0.0):
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
        
        key_map = {'train': 'train_data', 'val': 'val_data', 'test': 'test_data'}
        if mode not in key_map:
            raise ValueError(f"Invalid mode: {mode}")
        self.patients = all_data[key_map[mode]]

        self.samples = []
        for pid in self.patients:
            img_dir = os.path.join(self.filepath, 'img', pid)
            mask_dir = os.path.join(self.filepath, 'gt_lv', pid)

            if not (os.path.isdir(img_dir) and os.path.isdir(mask_dir)):
                continue

            img_list = sorted(os.listdir(img_dir), key=sort_by_number)
            
            if len(img_list) < self.seq_length:
                continue

            self.samples.append({
                'patient_id': pid,
                'img_dir': img_dir,
                'mask_dir': mask_dir,
                'img_list': img_list,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_dir = sample['img_dir']
        mask_dir = sample['mask_dir']
        img_list = sample['img_list']
        
        total_frames = len(img_list)
        if total_frames > self.seq_length:
            start_frame_idx = random.randint(0, total_frames - self.seq_length)
        else:
            start_frame_idx = 0
            
        sampled_names = img_list[start_frame_idx : start_frame_idx + self.seq_length]

        imgs_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)
        masks_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)

        for i, img_name in enumerate(sampled_names):
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, img_name)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                if img.shape != (self.size, self.size):
                    img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                imgs_np[i] = img
            
            if os.path.isfile(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != (self.size, self.size):
                        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    masks_np[i] = (mask == 1).astype(np.uint8)

        frames_t = torch.from_numpy(imgs_np).float().unsqueeze(1) / 255.0
        masks_t = torch.from_numpy(masks_np).long().unsqueeze(1)

        if self.mode == 'train':
            if random.random() < 0.5:
                frames_t = TF.hflip(frames_t)
                masks_t = TF.hflip(masks_t)
            
            angle = random.uniform(-10, 10)
            if angle != 0:
                frames_t = TF.rotate(frames_t, angle, interpolation=TF.InterpolationMode.BILINEAR)
                masks_t = TF.rotate(masks_t, angle, interpolation=TF.InterpolationMode.NEAREST)

        info = {
            'name': sample['patient_id'],
            'frames': sampled_names,
            'num_objects': 0
        }

        cls_gt = torch.zeros_like(masks_t)
        first_frame_gt = torch.zeros((1, self.max_num_obj, self.size, self.size), dtype=torch.long)
        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)

        if masks_t[0].max() > 0:
            selector[0] = 1.0
            cls_gt = masks_t.clone()
            first_frame_gt[0, 0] = masks_t[0, 0]
            info['num_objects'] = 1
        else:
            info['num_objects'] = 0

        data = {
            'rgb': frames_t,
            'ff_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data