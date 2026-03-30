# CVAT YOLO Satellite Object Detection

End-to-end object detection pipeline using CVAT annotations and YOLOv8 for detecting objects in satellite imagery.


## Project Overview

This project demonstrates a complete computer vision workflow:

1. Manual annotation using CVAT  
2. Dataset preparation and splitting  
3. Training a YOLOv8 model  
4. Evaluation and performance analysis  

The goal is to build a reproducible pipeline for object detection on satellite images.


## Dataset

### Source
SkySeaLand Satellite Object Detection Dataset  
https://data.mendeley.com/datasets/d42n3cp86p/3

### Classes
- airplane
- boat
- ship
- car

### Notes
- A subset of 350 images was selected  
- Images were manually annotated in CVAT (YOLO format)  
- The dataset is not included in this repository  


## Pipeline 

The project follows a structured end-to-end workflow:

1. **Annotation (CVAT)**
   - Manual bounding box annotation
   - Export in YOLO format

2. **Data Preparation**
   - Automated train/val/test split (80/10/10)
   - Directory structure creation
   - `dataset.yaml` generation

3. **Training (YOLOv8)**
   - Model: YOLOv8n (lightweight baseline)
   - Image size: 640
   - Batch size: 16
   - Epochs: 20

4. **Evaluation**
   - Precision-Recall curve
   - F1-confidence analysis
   - Confusion matrix
   - Qualitative prediction inspection

## Project Structure

```text
CVAT-YOLO-Satellite-Object-Detection/
│
├── data/
│   ├── README.md
│   └── sample/
│
├── outputs/
│   └── weights/
│       ├── best.pt
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── f1_curve.png
│   ├── gt_example_1.jpg
│   ├── gt_example_2.jpg
│   ├── pr_curve.png
│   ├── pred_example_1.jpg
│   └── pred_example_2.jpg
│
├── src/
│   ├── pipeline.py
│   ├── train.py
│   └── evaluate.py
│
├── LICENSE
├── README.md
└── requirements.txt
```


## Results

### Quantitative Performance

| Class     | mAP@0.5 |
|----------|--------|
| airplane | 0.881 |
| boat     | 0.689 |
| ship     | 0.648 |
| car      | 0.823 |
| **Overall** | **0.760** |

- Best overall F1-score: **0.72 at confidence ≈ 0.27**

### Performance Curves

<p align="center">
  <img src="https://github.com/IzaKam13/cvat-yolo-satellite-object-detection/blob/main/outputs/f1_curve.png" width="49%">
   <img src="https://github.com/IzaKam13/cvat-yolo-satellite-object-detection/blob/main/outputs/pr_curve.png" width="49%">
</p>

- Airplanes and cars show strong performance (high precision and recall)
- Boats and ships are more challenging due to:
  - Similar visual appearance
  - Smaller object size
  - Dense object grouping


## Dataset Analysis

### Class Distribution

- airplane: 1046
- car: 890
- boat: 346
- ship: 242

Total objects: 2524

### Interpretation

- Dataset is **imbalanced**:
  - Airplanes dominate
  - Ships are underrepresented

Impact:
- Model performs best on airplanes
- Lower recall for ships due to fewer examples

## Key Insights

- Data quality (annotation consistency) strongly impacts performance
- Class imbalance affects minority classes (ships, boats)
- YOLOv8n provides strong baseline even with small dataset
- Confidence threshold tuning is critical (optimal ~0.27)
- Dense object scenes remain challenging (overlapping boxes)

## Installation

``` Bash
git clone https://github.com/IzaKam13/cvat-yolo-satellite-object-detection/
cd CVAT-YOLO-Satellite-Object-Detection
pip install -r requirements.txt
```

## Future Improvements

- Increase dataset size (especially ships and boats)
- Apply data augmentation (rotation, scaling, noise)
- Try larger models (YOLOv8m / YOLOv8l)
- Hyperparameter tuning (learning rate, batch size)
- Use class balancing techniques
- Improve annotation consistency
- Add tracking or segmentation (Mask R-CNN / YOLOv8-seg)

## Why This Project Matters

This project demonstrates practical experience with:

- Real-world annotation tools (CVAT)
- Dataset engineering and preprocessing
- Training deep learning models (YOLOv8)
- Model evaluation and interpretation
- Handling class imbalance and noisy data

It reflects a production-oriented computer vision workflow.

## License
This project uses external dataset sources (SkySeaLand Satellite Object Detection Dataset, https://data.mendeley.com/datasets/d42n3cp86p/3). Please refer to the dataset provider for licensing details.
