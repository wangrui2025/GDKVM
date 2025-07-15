import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize

def visualize_sequence(rgb_seq, cls_gt_seq, out_dict, run_path, batch_idx_str):
    """
    Arranges visualizations of all frames in a sequence into a single static image.
    For long sequences (>10 frames), uses a grid layout instead of a single row.
    """
    logits_keys = sorted(
        [k for k in out_dict.keys() if k.startswith('logits_')],
        key=lambda x: int(x.split('_')[-1])
    )

    num_frames = len(logits_keys)
    if num_frames == 0:
        print("No frames to visualize.")
        return

    # Use a grid layout for long sequences
    if num_frames > 10:
        # Calculate grid dimensions: max 10 frames per row
        frames_per_row = min(10, num_frames)
        num_rows_per_type = (num_frames + frames_per_row - 1) // frames_per_row # Ceiling division
        total_rows = num_rows_per_type * 3  # 3 types of visualizations

        fig, axs = plt.subplots(total_rows, frames_per_row, figsize=(2.5 * frames_per_row, 3 * total_rows))
        if total_rows == 1:
            axs = np.expand_dims(axs, axis=0)
        if frames_per_row == 1:
            axs = np.expand_dims(axs, axis=1)
    else:
        # Original single-row layout
        fig, axs = plt.subplots(3, num_frames, figsize=(4 * num_frames, 12))
        if num_frames == 1:
            axs = np.expand_dims(axs, axis=1)

    fig.suptitle(f'Sequence Visualization: {batch_idx_str} ({num_frames} frames)', fontsize=20)
    norm = Normalize(vmin=0, vmax=1)

    for t in range(num_frames):
        key = logits_keys[t]
        rgb_frame = rgb_seq[t, 0]
        gt_frame = cls_gt_seq[t, 0]
        logits_frame = out_dict[key][0]
        if logits_frame.shape[0] == 2:
            logits_frame = logits_frame[1:2, :, :]
        prob_frame = torch.sigmoid(logits_frame).cpu().numpy().squeeze()
        pred_frame = (prob_frame > 0.5).astype(np.uint8)
        overlap_frame = np.logical_and(gt_frame, pred_frame)

        if num_frames > 10:
            # Grid layout: calculate position in the grid for the current frame
            frames_per_row = min(10, num_frames)
            frame_row = t // frames_per_row
            frame_col = t % frames_per_row

            # Original image row
            orig_row = frame_row
            axs[orig_row, frame_col].imshow(rgb_frame, cmap='gray')
            axs[orig_row, frame_col].set_title(f"Frame {t}", fontsize=10)
            axs[orig_row, frame_col].axis('off')

            # Segmentation overlay row
            overlay_row = frame_row + num_rows_per_type
            axs[overlay_row, frame_col].imshow(rgb_frame, cmap='gray')
            axs[overlay_row, frame_col].imshow(np.ma.masked_where(gt_frame == 0, np.ones_like(gt_frame)), cmap=ListedColormap([(0, 1, 0, 0.6)]))
            axs[overlay_row, frame_col].imshow(np.ma.masked_where(pred_frame == 0, np.ones_like(pred_frame)), cmap=ListedColormap([(1, 0, 0, 0.6)]))
            axs[overlay_row, frame_col].imshow(np.ma.masked_where(overlap_frame == 0, np.ones_like(overlap_frame)), cmap=ListedColormap([(1, 1, 0, 0.8)]))
            axs[overlay_row, frame_col].axis('off')

            # Heatmap row
            heatmap_row = frame_row + 2 * num_rows_per_type
            axs[heatmap_row, frame_col].imshow(rgb_frame, cmap='gray')
            axs[heatmap_row, frame_col].imshow(prob_frame, cmap='jet', alpha=0.5, interpolation='nearest', norm=norm)
            axs[heatmap_row, frame_col].axis('off')

        else:
            # Original single-row layout
            # Row 0: Original Image
            axs[0, t].imshow(rgb_frame, cmap='gray')
            axs[0, t].set_title(f"Frame {t}")
            axs[0, t].axis('off')

            # Row 1: Segmentation Overlay
            axs[1, t].imshow(rgb_frame, cmap='gray')
            axs[1, t].imshow(np.ma.masked_where(gt_frame == 0, np.ones_like(gt_frame)), cmap=ListedColormap([(0, 1, 0, 0.6)]))
            axs[1, t].imshow(np.ma.masked_where(pred_frame == 0, np.ones_like(pred_frame)), cmap=ListedColormap([(1, 0, 0, 0.6)]))
            axs[1, t].imshow(np.ma.masked_where(overlap_frame == 0, np.ones_like(overlap_frame)), cmap=ListedColormap([(1, 1, 0, 0.8)]))
            axs[1, t].axis('off')

            # Row 2: Heatmap
            axs[2, t].imshow(rgb_frame, cmap='gray')
            axs[2, t].imshow(prob_frame, cmap='jet', alpha=0.5, interpolation='nearest', norm=norm)
            axs[2, t].axis('off')

    # Set Y-axis labels
    if num_frames > 10:
        # Set labels for the grid layout
        for i in range(num_rows_per_type):
            if i * frames_per_row < num_frames:
                axs[i, 0].set_ylabel("Original", fontsize=14, rotation=90, labelpad=20)
            if (i + num_rows_per_type) * frames_per_row < total_rows:
                axs[i + num_rows_per_type, 0].set_ylabel("Overlay", fontsize=14, rotation=90, labelpad=20)
            if (i + 2 * num_rows_per_type) * frames_per_row < total_rows:
                axs[i + 2 * num_rows_per_type, 0].set_ylabel("Heatmap", fontsize=14, rotation=90, labelpad=20)

        # Hide unused subplots
        for row in range(total_rows):
            for col in range(frames_per_row):
                frame_idx = (row % num_rows_per_type) * frames_per_row + col
                if frame_idx >= num_frames:
                    axs[row, col].axis('off')
    else:
        # Original single-row layout labels
        axs[0, 0].set_ylabel("Original", fontsize=16, rotation=90, labelpad=20)
        axs[1, 0].set_ylabel("Overlay", fontsize=16, rotation=90, labelpad=20)
        axs[2, 0].set_ylabel("Heatmap", fontsize=16, rotation=90, labelpad=20)

    # Adjust subplot spacing
    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.95, bottom=0.05, left=0.08, right=0.95)

    save_path = os.path.join(run_path, f"{batch_idx_str}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Successfully saved sequence visualization with {num_frames} frames to {save_path}")
