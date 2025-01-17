import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def calculate_iou(pred, target):
    """计算IoU"""
    intersection = np.logical_and(target, pred)
    union = np.logical_or(target, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def calculate_metrics(predictions, targets):
    """计算Precision, Recall, F1和IoU"""
    # 将预测和目标展平为1D数组
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    # 计算Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_flat, 
        pred_flat, 
        average='binary'
    )
    
    # 计算IoU
    iou = calculate_iou(predictions, targets)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    } 