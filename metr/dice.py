import numpy as np
from tqdm import tqdm
import torch

def dice_coefficient(pred, gt, smooth=1e-5):
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1

    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    # if (pred.sum() + gt.sum()) == 0:
    #     return 1
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (unionset + smooth)
    return dice.sum() / N

# def sespiou_coefficient(pred, gt, all=False, smooth=1e-5):
def sespiou_coefficient(pred, gt, all=True, smooth=1e-5):
    N = gt.shape[0]
    pred[pred >= 1] = 1
    gt[gt >= 1] = 1
    pred_flat = pred.reshape(N, -1)
    gt_flat = gt.reshape(N, -1)
    #pred_flat = pred.view(N, -1)
    #gt_flat = gt.view(N, -1)

    TP = (pred_flat * gt_flat).sum(1)
    FN = gt_flat.sum(1) - TP
    pred_flat_no = (pred_flat + 1) % 2
    gt_flat_no = (gt_flat + 1) % 2
    TN = (pred_flat_no * gt_flat_no).sum(1)
    FP = pred_flat.sum(1) - TP

    SE = (TP + smooth) / (TP + FN + smooth)
    SP = (TN + smooth) / (FP + TN + smooth)
    IOU = (TP + smooth) / (FP + TP + FN + smooth)
    Acc = (TP + TN + smooth)/(TP + FP + FN + TN + smooth)

    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1 = 2*Precision*Recall/(Recall + Precision +smooth)

    return IOU.sum() / N, Acc.sum()/N, SE.sum() / N, SP.sum() / N, F1.sum()/N, Precision.sum()/N, Recall.sum()/N
    
    if all:
        return IOU.sum() / N, Acc.sum()/N, SE.sum() / N, SP.sum() / N, F1.sum()/N, Precision.sum()/N, Recall.sum()/N
    else:
        return IOU.sum() / N, Acc.sum()/N, SE.sum() / N, SP.sum() / N
