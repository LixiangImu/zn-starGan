import os
import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score

def calculate_iou(pred, target, num_classes):
    iou_list = []
    for cls in range(num_classes):
        intersection = np.logical_and(pred == cls, target == cls)
        union = np.logical_or(pred == cls, target == cls)
        if np.sum(union) == 0:
            iou_list.append(0)
        else:
            iou_list.append(np.sum(intersection) / np.sum(union))
    return np.mean(iou_list)

def evaluate(pred_dir, target_dir, num_classes):
    pred_images = sorted(os.listdir(pred_dir))
    target_images = sorted(os.listdir(target_dir))

    all_preds = []
    all_targets = []

    for pred_img, target_img in zip(pred_images, target_images):
        pred = cv2.imread(os.path.join(pred_dir, pred_img), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join(target_dir, target_img), cv2.IMREAD_GRAYSCALE)

        all_preds.append(pred.flatten())
        all_targets.append(target.flatten())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    iou = calculate_iou(all_preds, all_targets, num_classes)

    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'IoU: {iou:.4f}')

if __name__ == "__main__":
    pred_dir = 'path/to/generated/images'  # 生成的图像目录
    target_dir = 'path/to/ground/truth'     # 真实标签目录
    num_classes = 2  # 根据您的数据集设置类别数量

    evaluate(pred_dir, target_dir, num_classes)