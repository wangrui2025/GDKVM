import random
import numpy as np
import torch

# im_mean = (124, 116, 104)
im_mean = (0.5, 0.5, 0.5)

def reseed(seed):
    """仿照 VOS 里的做法，让每个增广操作使用相同随机种子以保证可重现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, l in enumerate(labels):
        Ms[ni] = (masks == l).astype(np.uint8)

    return Ms


def sort_by_number(filename: str) -> int:
    """
    根据文件名中的数字进行排序的辅助函数。
    例如 '3.png' -> 3, '10.png' -> 10。
    若提取不到数字，则返回 -1。
    """
    # 把文件名用 '.' 分割，再判断哪一部分是纯数字
    # 也可以考虑用正则表达式来提取数字
    for part in filename.split('.'):
        if part.isdigit():
            return int(part)
    return -1

def correct_dims(*images):
    """
    将 (H, W) 的图像扩展到 (H, W, 1)，以便后续 Albumentations 能处理通道维度。
    如果只传入一个图像，则返回单个处理后的图像；若传入多个，则返回列表。
    """
    outputs = []
    for img in images:
        if len(img.shape) == 2:  # (H, W)
            img = np.expand_dims(img, axis=2)  # => (H, W, 1)
        outputs.append(img)
    if len(outputs) == 1:
        return outputs[0]
    return outputs
