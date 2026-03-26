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



## Project Structure

```text
CVAT-YOLO-Satellite-Object-Detection/
│
├── data/
│   ├── README.md
│   └── sample/
│
├── outputs/
│   └── class_distribution.png
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


## Dataset Analysis


## Key Insights


## Installation

``` Bash
git clone https://github.com/IzaKam13/cvat-yolo-satellite-object-detection/
cd CVAT-YOLO-Satellite-Object-Detection
pip install -r requirements.txt
```

## Future Improvements


## License
This project uses external dataset sources (SkySeaLand Satellite Object Detection Dataset, https://data.mendeley.com/datasets/d42n3cp86p/3). Please refer to the dataset provider for licensing details.
