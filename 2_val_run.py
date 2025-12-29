#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   2_val_run.py
@Time    :   2025/10/31 13:12:14
@Author  :   Li Zhuqiang
@Version :   1.0
@Contact :   lizq@jlu.edu.cn
@License :   (C)Copyright 2021-2024, Jilin University
@Desc    :   None
'''
import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

model = YOLO("runs/obb/train/weights/best.pt")
metrics=model.val(data="data/mixoldata.yaml",batch=4)
print(metrics.box.map)