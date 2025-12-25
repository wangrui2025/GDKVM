import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from pathlib import Path

def visualize_sequence(rgb_seq, cls_gt_seq, out_dict, run_path, batch_idx_str, iteration=None, epoch=None, patient_id=None, channel_type=None, mode='val'):
    logits_keys = sorted(
        [k for k in out_dict.keys() if k.startswith('logits_')],
        key=lambda x: int(x.split('_')[-1])
    )

    num_frames = len(logits_keys)
    if num_frames == 0:
        print("No frames to visualize.")
        return

    if num_frames > 15:
        frames_per_row = min(15, num_frames)
        num_rows_per_type = (num_frames + frames_per_row - 1) // frames_per_row
        total_rows = num_rows_per_type * 3
        
        fig, axs = plt.subplots(total_rows, frames_per_row, figsize=(2.5 * frames_per_row, 3 * total_rows), squeeze=False)
    else:
        fig, axs = plt.subplots(3, num_frames, figsize=(4 * num_frames, 12), squeeze=False)

    norm = Normalize(vmin=0, vmax=1)
    cmap_gt = ListedColormap([(0, 1, 0, 0.6)])
    cmap_pred = ListedColormap([(1, 0, 0, 0.6)])
    cmap_overlap = ListedColormap([(1, 1, 0, 0.8)])

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

        # Determine axes
        if num_frames > 15:
            frame_row = t // frames_per_row
            frame_col = t % frames_per_row
            ax_orig = axs[frame_row, frame_col]
            ax_overlay = axs[frame_row + num_rows_per_type, frame_col]
            ax_heatmap = axs[frame_row + 2 * num_rows_per_type, frame_col]
        else:
            ax_orig = axs[0, t]
            ax_overlay = axs[1, t]
            ax_heatmap = axs[2, t]

        # Plot Original
        ax_orig.imshow(rgb_frame, cmap='gray')
        ax_orig.set_title(f"Frame {t}", fontsize=10 if num_frames > 15 else 12)
        ax_orig.axis('off')

        # Plot Overlay
        ax_overlay.imshow(rgb_frame, cmap='gray')
        ax_overlay.imshow(np.ma.masked_where(gt_frame == 0, np.ones_like(gt_frame)), cmap=cmap_gt)
        ax_overlay.imshow(np.ma.masked_where(pred_frame == 0, np.ones_like(pred_frame)), cmap=cmap_pred)
        ax_overlay.imshow(np.ma.masked_where(overlap_frame == 0, np.ones_like(overlap_frame)), cmap=cmap_overlap)
        ax_overlay.axis('off')

        # Plot Heatmap
        ax_heatmap.imshow(rgb_frame, cmap='gray')
        ax_heatmap.imshow(prob_frame, cmap='jet', alpha=0.5, interpolation='nearest', norm=norm)
        ax_heatmap.axis('off')

    if num_frames > 15:
        for i in range(num_rows_per_type):
            if i * frames_per_row < num_frames:
                axs[i, 0].set_ylabel("Original", fontsize=14, rotation=90, labelpad=20)
            if (i + num_rows_per_type) * frames_per_row < total_rows:
                axs[i + num_rows_per_type, 0].set_ylabel("Overlay", fontsize=14, rotation=90, labelpad=20)
            if (i + 2 * num_rows_per_type) * frames_per_row < total_rows:
                axs[i + 2 * num_rows_per_type, 0].set_ylabel("Heatmap", fontsize=14, rotation=90, labelpad=20)
        
        for row in range(total_rows):
            for col in range(frames_per_row):
                frame_idx = (row % num_rows_per_type) * frames_per_row + col
                if frame_idx >= num_frames:
                    axs[row, col].axis('off')
    else:
        axs[0, 0].set_ylabel("Original", fontsize=16, rotation=90, labelpad=20)
        axs[1, 0].set_ylabel("Overlay", fontsize=16, rotation=90, labelpad=20)
        axs[2, 0].set_ylabel("Heatmap", fontsize=16, rotation=90, labelpad=20)
    
    fig.subplots_adjust(wspace=0.02, hspace=0.005, top=0.95, bottom=0.05, left=0.08, right=0.95)
    
    filename_parts = []
    
    if iteration is not None:
        filename_parts.append(f"It_{iteration:04d}")
    
    if epoch is not None:
        filename_parts.append(f"E_{epoch:02d}")
    
    filename_parts.append(mode)
    
    if patient_id is not None:
        filename_parts.append(str(patient_id))
    
    filename = "_".join(filename_parts) + ".png"
    
    save_path = Path(run_path) / filename
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Successfully saved sequence visualization with {num_frames} frames to {save_path}")