import os 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from thop import profile
import torch

model = YOLO("yaml/DLANet_yolo11l-obb.yaml").model
model.eval()
dummy = torch.randn(1, 6, 640, 640)

flops, params = profile(model, inputs=(dummy,), verbose=False)

print(f"FLOPs: {flops/1e9:.2f} GFLOPs")