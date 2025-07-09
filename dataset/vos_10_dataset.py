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
    """
    data = {
        'rgb': images,             # [T, C, H, W], float in [0,1]
        'first_frame_gt': ff_gt,   # [1, max_num_obj, H, W], long
        'cls_gt': cls_gt,          # [T, 1, H, W], long
        'selector': selector,      # [max_num_obj], float
        'info': info,
    }
    """
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
        if mode == 'train':
            self.patients = all_data['train_data']
        elif mode == 'val':
            self.patients = all_data['val_data']
        elif mode == 'test':
            self.patients = all_data['test_data']
        else:
            raise ValueError(f"Invalid mode: {mode}")


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


        if self.mode == 'train':
            self.seq_transform_img = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
            ])
        else:
            self.seq_transform_img = transforms.Compose([
            ])


        self.frame_transform_img = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),  # => [0,1]
        ])


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

        frames = []
        masks = []

        for i in range(T):
            img_name = img_list[i]
            img_path = os.path.join(img_dir, img_name)

            if i == 0 or i == T - 1:
                mask_path = os.path.join(mask_dir, img_name)
                if not os.path.isfile(mask_path):
                    log.warning(f"mask not found at {mask_path}!")

                    mask_img = np.zeros((256, 256), dtype=np.uint8)
                else:
                    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask_img is None:
                        log.warning(f"Failed to read mask from {mask_path}, using zeros.")
                        mask_img = np.zeros((256, 256), dtype=np.uint8)
            else:
                mask_img = np.zeros((256, 256), dtype=np.uint8)

            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                log.warning(f"Failed to read image from {img_path}, using zeros.")
                image = np.zeros((256, 256), dtype=np.uint8)

            pil_img = to_pil_image(image)
            pil_mask = to_pil_image(mask_img)

            frames.append(pil_img)
            masks.append(pil_mask)

        seq_seed = random.randint(0, 99999)

        transformed_frames = []
        transformed_masks = []

        for i in range(T):
            reseed(seq_seed)
            f_img = self.seq_transform_img(frames[i])
            transformed_frames.append(f_img)

            reseed(seq_seed)
            f_mask = self.seq_transform_img(masks[i])
            transformed_masks.append(f_mask)


        final_imgs = []
        final_masks = []
        for i in range(T):
            img_t = self.frame_transform_img(transformed_frames[i])  # => [C,H,W], float in [0,1]
            # mask
            mask_pil = self.frame_transform_mask(transformed_masks[i])
            mask_np = np.array(mask_pil, dtype=np.uint8)
            mask_np[mask_np != 1] = 0 
            # mask_np[mask_np > 0] = 1 
            mask_t = torch.from_numpy(mask_np).long().unsqueeze(0)  # => shape [1,H,W]

            final_imgs.append(img_t)
            final_masks.append(mask_t)

        final_imgs_t = torch.stack(final_imgs, dim=0)   # => [T, C, H, W]
        final_masks_t = torch.stack(final_masks, dim=0) # => [T, 1, H, W]

        info = {
            'name': patient_id,
            'frames': [img_list[i] for i in range(T)]
        }

        first_frame_labels = final_masks_t[0].unique()
        first_frame_labels = [v.item() for v in first_frame_labels if v.item() != 0]
        target_objects = first_frame_labels[:self.max_num_obj]


        cls_gt = torch.zeros_like(final_masks_t)  # dtype long, shape [T,1,H,W]


        first_frame_gt = torch.zeros(
            (1, self.max_num_obj, self.size, self.size), dtype=torch.long
        )

        for i, l in enumerate(target_objects):
          
            this_mask = (final_masks_t == l)

            cls_gt[this_mask] = i + 1

            first_frame_gt[0, i] = this_mask[0, 0].long()

        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)
        for i in range(len(target_objects)):
            selector[i] = 1.0

        info['num_objects'] = len(target_objects)

        data = {
            'rgb'      : final_imgs_t,    # [T,C,H,W], float in [0,1]
            'ff_gt'    : first_frame_gt,  # [T=1, num_objects, H, W]
            'cls_gt'   : cls_gt,          # [T,1,H,W], long
            'selector' : selector,        # [1]
            'info'     : info,
        }

        return data
