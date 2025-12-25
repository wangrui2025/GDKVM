# Visualization Utilities

This module provides functionality for visualizing sequence segmentation results.

## `vis_0730.py`

This script contains the `visualize_sequence` function, which generates a composite visualization of image sequences, ground truth masks, and predicted segmentation maps.

### Features

- **Adaptive Layout**:
  - **Grid Layout**: For long sequences (>15 frames), frames are arranged in a grid to maintain readability.
  - **Single Row Layout**: For shorter sequences, frames are displayed in a single row.
- **Visualization Components**:
  - **Original Image**: The raw input frame.
  - **Overlay**: A composite view showing the original image with ground truth (green), prediction (red), and overlap (yellow) masks.
  - **Heatmap**: A probability heatmap of the model's output overlaid on the original image.
- **Output**: Saves the visualization as a high-resolution PNG file.

### Usage

```python
from vis.vis0730_1100 import visualize_sequence

visualize_sequence(
    rgb_seq=...,        # Input RGB frames
    cls_gt_seq=...,     # Ground truth masks
    out_dict=...,       # Model output dictionary containing logits
    run_path=...,       # Directory to save the output
    batch_idx_str=...,  # Batch identifier
    iteration=...,      # Current iteration (optional)
    epoch=...,          # Current epoch (optional)
    patient_id=...,     # Patient ID (optional)
    mode='val'          # Mode (e.g., 'val', 'test')
)
```
