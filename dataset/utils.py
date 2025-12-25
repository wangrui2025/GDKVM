import random
import numpy as np
import torch

# im_mean = (124, 116, 104)
im_mean = (0.5, 0.5, 0.5)

def reseed(seed):
    """
    Reseeds random number generators to ensure deterministic behavior for synchronized augmentations, 
    following standard protocols in Video Object Segmentation (VOS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]), dtype=np.uint8)

    for ni, label_value in enumerate(labels):
        Ms[ni] = (masks == label_value).astype(np.uint8)

    return Ms


def sort_by_number(filename: str) -> int:
    """
    Auxiliary function to sort filenames based on their numeric components.
    Example: '3.png' -> 3, '10.png' -> 10.
    Returns -1 if no numeric component is found.
    """
    # Split the filename by '.' and identify the purely numeric part.
    # Alternatively, regular expressions could be used for digit extraction.
    for part in filename.split('.'):
        if part.isdigit():
            return int(part)
    return -1

def correct_dims(*images):
    """
    Expands image dimensions from (H, W) to (H, W, 1) to ensure compatibility with 
    Albumentations' channel processing requirements.
    
    Returns:
        Single image if one input is provided; otherwise, returns a list of processed images.
    """
    outputs = []
    for img in images:
        if len(img.shape) == 2:  # (H, W)
            img = np.expand_dims(img, axis=2)  # => (H, W, 1)
        outputs.append(img)
    if len(outputs) == 1:
        return outputs[0]
    return outputs