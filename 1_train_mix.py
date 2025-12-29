#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   1_train_mix.py
@Time    :   2025/10/31 16:11:59
@Author  :   Li Zhuqiang
@Version :   1.0
@Contact :   lizq@jlu.edu.cn
@License :   (C)Copyright 2021-2024, Jilin University
@Desc    :   None
'''

# here put the import lib
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

# Load a pretrained model
model = YOLO("yaml/DLANet_yolo11l-obb.yaml")

# Train the model
results = model.train(data="data/mixoldata.yaml", batch=4,epochs=50, imgsz=640,patience=40)
