import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class EchoDataset(Dataset):
    def __init__(self, filepath: str, mode: str = 'train', seq_length=10, max_num_obj=1, size=128, merge_probability=0.0):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.seq_length = seq_length
        self.max_num_obj = max_num_obj
        self.size = size
        self.merge_probability = merge_probability

        self.img_root = os.path.join(filepath, mode, 'img')
        self.label_root = os.path.join(filepath, mode, 'label')
        
        self.samples = []
        
        if os.path.isdir(self.img_root) and os.path.isdir(self.label_root):
            subfolders = sorted(os.listdir(self.img_root))
            
            for subfolder in subfolders:
                img_folder = os.path.join(self.img_root, subfolder)
                label_folder = os.path.join(self.label_root, subfolder)
                
                if os.path.isdir(img_folder) and os.path.isdir(label_folder):
                    img_files = sorted(os.listdir(img_folder))
                    label_files = sorted(os.listdir(label_folder))

                    if len(img_files) == 10 and len(label_files) == 2:
                        self.samples.append({
                            'subfolder': subfolder,
                            'img_folder': img_folder,
                            'label_folder': label_folder,
                            'img_files': img_files,
                            'label_files': label_files,
                        })
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img_folder = sample['img_folder']
        label_folder = sample['label_folder']
        img_files = sample['img_files']
        label_files = sample['label_files']

        imgs_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)
        masks_np = np.zeros((self.seq_length, self.size, self.size), dtype=np.uint8)

        for i in range(self.seq_length):
            img_path = os.path.join(img_folder, img_files[i])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                if img.shape != (self.size, self.size):
                    img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                imgs_np[i] = img
            
            mask_path = None
            if i == 0:
                mask_path = os.path.join(label_folder, label_files[0])
            elif i == self.seq_length - 1:
                mask_path = os.path.join(label_folder, label_files[1])

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    if mask.shape != (self.size, self.size):
                        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                    masks_np[i] = (mask == 1).astype(np.uint8)

        frames_t = torch.from_numpy(imgs_np).float().unsqueeze(1) / 255.0
        masks_t = torch.from_numpy(masks_np).long().unsqueeze(1)

        info = {
            'name': sample['subfolder'],
            'frames': img_files,
            'num_objects': 0
        }

        cls_gt = torch.zeros_like(masks_t)
        first_frame_gt = torch.zeros((1, self.max_num_obj, self.size, self.size), dtype=torch.long)
        selector = torch.zeros(self.max_num_obj, dtype=torch.float32)

        if masks_t[0].max() > 0:
            info['num_objects'] = 1
            selector[0] = 1.0
            
            cls_gt = masks_t.clone()
            first_frame_gt[0, 0] = masks_t[0, 0]

        data = {
            'rgb': frames_t,
            'ff_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data