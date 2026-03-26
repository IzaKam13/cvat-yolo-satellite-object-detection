# CVAT YOLO Satellite Object Detection


## Dataset Source

This project is based on the SkySeaLand Satellite Object Detection Dataset  
(Source: Mendeley Data, https://data.mendeley.com/datasets/d42n3cp86p/3).

The original dataset contains four object classes:
- airplane
- boat
- ship
- car

## Dataset Preparation 

- The full dataset is not included in this repository.
- Although the original dataset contains annotations, the images were manually re-annotated in CVAT in order to practice annotation workflow and build a custom object detection pipeline.
- For this project, a subset of 350 images was selected, manually annotated using CVAT, and prepared for YOLO object detection.
- Use `pipeline.py` to prepare the dataset for training.
- The `pipeline.py` script converts CVAT-exported YOLO annotation files into a training-ready dataset structure:
    - train / val / test split,
    - images/labels directory layout,
    - dataset.yaml generation.
- Input expected by the script:
    - image files,
    - corresponding YOLO .txt annotation files exported from CVAT.
