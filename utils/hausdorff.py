import torch

def hausdorff_distance_gpu(pred, target, distance="euclidean"):
    """
    计算两个二值图像分割结果的 Hausdorff 距离，在 GPU 上进行计算。

    参数:
    - pred (torch.Tensor): 预测分割结果，二值化的图像，形状为 (B, H, W)
    - target (torch.Tensor): 目标分割结果，二值化的图像，形状为 (B, H, W)
    - distance (str): "euclidean" 或 "chessboard" 距离类型，默认为 "euclidean"

    返回:
    - hd (float): 计算得到的 Hausdorff 距离
    """

    assert pred.shape == target.shape, "预测和目标的形状必须一致"
    
    # 获取预测和目标中为1的像素位置
    pred_points = torch.nonzero(pred).float()  # shape: (num_points, 2)
    target_points = torch.nonzero(target).float()  # shape: (num_points, 2)
    
    if pred_points.size(0) == 0 or target_points.size(0) == 0:
        return torch.tensor(0.0, device=pred.device)  # 如果没有1像素，返回0

    # 计算点到点的距离矩阵
    pred_expanded = pred_points.unsqueeze(1)  # (num_pred_points, 1, 2)
    target_expanded = target_points.unsqueeze(0)  # (1, num_target_points, 2)
    
    # 计算每个预测点到每个目标点的距离
    if distance == "euclidean":
        dist_matrix = torch.sqrt(torch.sum((pred_expanded - target_expanded) ** 2, dim=2))  # (num_pred_points, num_target_points)
    elif distance == "chessboard":
        dist_matrix = torch.max(torch.abs(pred_expanded - target_expanded), dim=2)[0]  # (num_pred_points, num_target_points)
    else:
        raise ValueError("Invalid distance type. Use 'euclidean' or 'chessboard'.")

    # 计算 Hausdorff 距离
    hd_pred_to_target = torch.max(torch.min(dist_matrix, dim=1)[0])  # 每个预测点到目标点的最小距离中的最大值
    hd_target_to_pred = torch.max(torch.min(dist_matrix, dim=0)[0])  # 每个目标点到预测点的最小距离中的最大值

    hd = torch.max(hd_pred_to_target, hd_target_to_pred)
    return hd