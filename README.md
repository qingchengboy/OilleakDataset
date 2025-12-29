# OilleakDataset
 Oilleak is a dual-modal oil spill dataset collected by UAVs, featuring paired RGB and thermal-infrared images for object detection. We are in the process of organizing the dataset, code, and documentation. Thank you for your interest — updates to the repository will be available soon.

## 🖼️ Figures (displays some data in the dataset)
![Example 1](example/examples.png) 

## ⚙️ Usage Instructions
### 1. Environment Dependencies
 The provided requirements.txt file contains the dependencies for our conda environment; you can refer to it for necessary libraries. Alternatively, you can simply run：
 `pip install -r requirements.txt`
### 2. Data
**The OilleakDataset comprises registered RGB and TIR (Thermal Infrared) image pairs.**

Cloud drive link: https://drive.google.com/file/d/1nKYtcaOMtvh0zvcz4tcfedB1e_fhLSpO/view

(Please note that, due to the project owner's current requirements, we are only able to provide a partial dataset for this delivery (specifically, the data that has already been registered/aligned). We anticipate being able to upload the complete dataset in a subsequent update/delivery.)

The dataset structure is as follows:

* The `images` folder holds the **RGB images**.
* The `irimages` folder contains the corresponding **TIR images**.
* The `label` folder provides the **YOLO format label files**: `[class_id, x1, y1, x2, y2, x3, y3, x4, y4]`.

### 3. 🚀 Training
Please follow the steps below to configure your data and network structure:

1.  **Data Configuration (`data/mixoldata.yaml`)**
    * Please modify the path information within the configuration file `data/mixoldata.yaml` to match your actual data storage location.
    * **Note:** Ensure the paths correctly point to your dataset folders.

2.  **Network Architecture Adjustment (Optional)**
    * If you need to adjust the network architecture (e.g., modifying model depth, width, etc.), please refer to the following configuration file:
        * `yaml/DLANet_yolo11l-obb.yaml`
3.  **Begin Training**
     * **Note:**  The training strategy, such as epochs and batch size, can be adjusted according to the actual task.
     ```bash
     python 1_train_mix.py
     
### 4. 🔍 Inference
To assess model performance, use the following scripts:

* **Evaluate Validation Accuracy:**
    Run the script `python 2_val_run.py` to evaluate the accuracy of the validation set.

* **Visualize Predictions:**
    Run the script `python 3_vis_predict_obb.py` to visualize the effect of the predicted targets overlaid on the images.

## 📝 Citation

If you find the Oilleak Dataset or the associated code useful for your research, please cite our paper:

@Article{rs18010107,
AUTHOR = {Li, Zhuqiang and Zhen, Zhijun and Chen, Shengbo and Zhang, Liqiang and Cao, Lisai},
TITLE = {Dual-Level Attention Relearning for Cross-Modality Rotated Object Detection in UAV RGB–Thermal Imagery},
JOURNAL = {Remote Sensing},
VOLUME = {18},
YEAR = {2026},
NUMBER = {1},
ARTICLE-NUMBER = {107},
URL = {https://www.mdpi.com/2072-4292/18/1/107},
ISSN = {2072-4292},
DOI = {10.3390/rs18010107}
}


