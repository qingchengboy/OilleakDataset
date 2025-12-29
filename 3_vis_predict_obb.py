#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   3_vis_predict_obb.py
@Time    :   2025/04/25 13:53:47
@Author  :   Li Zhuqiang
@Version :   1.0
@Contact :   lizq@jlu.edu.cn
@License :   (C)Copyright 2021-2024, Jilin University
@Desc    :   None
'''

# here put the import lib
# 多模态有向边框推理
import cv2
import torch
import numpy as np
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
import os
import time

from ultralytics import YOLO


def xywhr2xyxyxyxy(center):
    # reference: https://github.com/ultralytics/ultralytics/blob/v8.1.0/ultralytics/utils/ops.py#L545
    is_numpy = isinstance(center, np.ndarray)
    cos, sin = (np.cos, np.sin) if is_numpy else (torch.cos, torch.sin)

    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1) if is_numpy else torch.cat(vec1, dim=-1)
    vec2 = np.concatenate(vec2, axis=-1) if is_numpy else torch.cat(vec2, dim=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2) if is_numpy else torch.stack([pt1, pt2, pt3, pt4], dim=-2)



def load_model(weights_path, device):
    if not os.path.exists(weights_path):
        print("Model weights not found!")
        exit()
    model = YOLO(weights_path).to(device)
    model.fuse()
    model.info(verbose=False)
    return model

def _get_covariance_matrix(obb):
    """
    计算旋转边界框的协方差矩阵。
    :param obb: 旋转边界框 (Oriented Bounding Box)，包含中心坐标、宽、高和旋转角度
    :return: 协方差矩阵的三个元素 a, b, c
    """
    widths = obb[..., 2] / 2  # 获取宽度的一半
    heights = obb[..., 3] / 2  # 获取高度的一半
    angles = obb[..., 4]  # 获取旋转角度
 
    cos_angle = np.cos(angles)  # 计算旋转角度的余弦值
    sin_angle = np.sin(angles)  # 计算旋转角度的正弦值
 
    # 计算协方差矩阵的三个元素 a, b, c
    a = (widths * cos_angle) ** 2 + (heights * sin_angle) ** 2
    b = (widths * sin_angle) ** 2 + (heights * cos_angle) ** 2
    c = widths * cos_angle * heights * sin_angle
 
    return a, b, c

def batch_probiou(obb1, obb2, eps=1e-7):
    """
    计算旋转边界框之间的 ProbIoU。
    :param obb1: 第一个旋转边界框集合
    :param obb2: 第二个旋转边界框集合
    :param eps: 防止除零的极小值
    :return: 两个旋转边界框之间的 ProbIoU
    """
    # 提取两个旋转边界框的中心坐标 (x, y)
    x1, y1 = obb1[..., 0], obb1[..., 1]
    x2, y2 = obb2[..., 0], obb2[..., 1]
 
    # 计算两个旋转边界框的协方差矩阵元素 a, b, c
    a1, b1, c1 = _get_covariance_matrix(obb1)
    a2, b2, c2 = _get_covariance_matrix(obb2)
 
    # 计算 ProbIoU 的三个部分 t1, t2, t3
    # t1 表示中心点位置差异的贡献
    t1 = ((a1[:, None] + a2) * (y1[:, None] - y2) ** 2 + (b1[:, None] + b2) * (x1[:, None] - x2) ** 2) / (
            (a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps) * 0.25
    # t2 表示旋转角度的耦合贡献
    t2 = ((c1[:, None] + c2) * (x2 - x1[:, None]) * (y1[:, None] - y2)) / (
            (a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2 + eps) * 0.5
    # t3 表示面积和形状之间的差异贡献
    t3 = np.log(((a1[:, None] + a2) * (b1[:, None] + b2) - (c1[:, None] + c2) ** 2) /
                (4 * np.sqrt((a1 * b1 - c1 ** 2)[:, None] * (a2 * b2 - c2 ** 2)) + eps) + eps) * 0.5
 
    # 计算 Bhattacharyya 距离 bd
    bd = np.clip(t1 + t2 + t3, eps, 100.0)  # 将 bd 限制在 [eps, 100.0] 范围内
 
    # 计算 ProbIoU 值 hd
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)  # 使用 Bhattacharyya 距离计算 hd
    return 1 - hd  # 返回 1 - hd，hd 越小表示相似度越高，1 - hd 即为 ProbIoU

def rotated_nms_with_probiou(boxes, scores, iou_threshold=0.5):
    order = scores.argsort()[::-1]  
    keep = []  
 
    while len(order) > 0:
        i = order[0]  
        keep.append(i) 
 
        if len(order) == 1: 
            break
 
        remaining_boxes = boxes[order[1:]] 
        iou_values = batch_probiou(boxes[i:i + 1], remaining_boxes).squeeze(0) 
 
        mask = iou_values < iou_threshold 
        order = order[1:][mask]  
 
    return keep  




def DLANet(rgb_image_path,ir_image_path, out_put,model,conf_sorce=0.3):
    
    image_name=os.path.basename(rgb_image_path)
    imgr = cv2.imread(rgb_image_path)
    irimg = cv2.imread(ir_image_path)
    img=np.concatenate((imgr,irimg),axis=2)
    result = model.predict(img,save=False,imgsz=640,visualize=False,conf=conf_sorce)#
    conf,cls, xywhr = result[0].obb.conf,result[0].obb.cls, result[0].obb.xywhr
    confs,classes, xywhr_ = conf.detach().cpu().numpy(),cls.detach().cpu().numpy(), xywhr.detach().cpu().numpy()
    
    result = rotated_nms_with_probiou(xywhr_, confs, iou_threshold=0.5)
    xywhr_=xywhr_[result]
    confs=confs[result]
    
    boxes   = xywhr2xyxyxyxy(np.array(xywhr_))
    
    names =['OilLeak']
    colors = [
        (255, 255, 0), 
    ]

    img=irimg
    
    for i, box in enumerate(boxes):
        confidence = confs[i]
        label = int(classes[i])
        color = colors[label]
        
        cv2.polylines(img[...,:3], [np.asarray(box, dtype=int)], True, color, 2)
        left, top = [int(b) for b in box[0]]
        cv2.polylines(imgr[...,:3], [np.asarray(box, dtype=int)], True, color, 2)

    cv2.imwrite(os.path.join(out_put,"our_ir_"+image_name), img)
    cv2.imwrite(os.path.join(out_put,"our_rgb_"+image_name), imgr)
    print("save done")


if __name__ == "__main__":
    output_path = 'output'    
    ir_image_dir = 'OLdatasets/irimages/val'       # 输入图像路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = load_model("runs/obb/train/weights/best.pt", device)
    start_time=time.time()
    
    for f_path in os.listdir(ir_image_dir):
        imagename=f_path

        ir_image_path=os.path.join(ir_image_dir,imagename)
        rgb_image_path= os.path.join("OLdatasets/images/val",imagename)
    
        
        DLANet(rgb_image_path,ir_image_path, output_path,model,conf_sorce=0.3)
    end_time=time.time()
    print("mean time:",(end_time-start_time)/len(os.listdir(ir_image_dir)))
    