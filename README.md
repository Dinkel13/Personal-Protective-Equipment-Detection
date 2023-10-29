
## Introduction
This documentary file provides an overview of a project involving the detection of Personal Protective Equipment (PPE) using computer vision. The project encompassed data collection, data labeling, and code implementation to achieve the best results. The project was conducted on a cloud-based machine, with specific configurations and parameter tuning.

## Data Collection

### Data Source

For this project, a diverse dataset of images containing individuals wearing different types of PPE was collected. The data source was a compressed ZIP file containing various subdirectories, each containing images of different PPE scenarios.

You can download the following files into the project directory:
https://universe.roboflow.com/ai-project-yolo/ppe-detection-q897z

## Data Preparation
### 1. Data Extraction: 
The first step was to extract the data from the ZIP file, obtaining a structured dataset.

### 2. Data Exploration: 
The dataset was examined to ensure it had a balance of positive (images with PPE) and negative (images without PPE) samples. Data preprocessing techniques like resizing, augmentation, and noise reduction were applied to improve data quality.
## Data Labeling

For the machine learning model to recognize PPE in images, the dataset needed to be labeled. The labeling process was facilitated using CVAT.ai, a Computer Vision Annotation Tool. The following steps were taken: The following steps were taken:

### 1. Data Annotation: 
The dataset was uploaded to CVAT.ai, and bounding boxes were drawn around instances of PPE in the images.

### 2. Labeling Quality Control: 
A quality control process was implemented to ensure accurate labeling and to address any discrepancies in the annotations.

### 3. Export Labels: 
Once the data was accurately labeled, the annotations were exported in a yolov8 format that could be easily integrated into the training pipeline.
## Code Implementation

The project utilized a cloud-based machine for its computational needs. The cloud machine was configured as follows:

### 1. Instance Type: 
An appropriate cloud machine instance type with adequate GPU resources was selected.

### 2. Operating System: 
A suitable operating system was chosen to support the project's requirements where I have selected Linux OS.

### 3. Software Installation: 
Necessary software libraries, frameworks, and dependencies were installed on the cloud machine.


## Setup for Code

1. Prepare a .yaml file named config.yaml :

```http
path: '/mnt/data' # dataset root dir
train: '/mnt/data/images'  # train images (relative to 'path')
val: '/mnt/data/images'  # val images (relative to 'path')

# Classes
names:
  0: No helmet
  1: No vest
  2: Person 
  3: helmet 
  4: vest  
```

2. Now make a python file named train.py :

```http
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=5000)  # train the model
```

### Run the Training

In the terminal, run the following command:

```python train.py```

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 700 consecutive epoches.





## Setup for Code

1. Prepare a .yaml file named config.yaml :

```http
path: '/mnt/data' # dataset root dir
train: '/mnt/data/images'  # train images (relative to 'path')
val: '/mnt/data/images'  # val images (relative to 'path')

# Classes
names:
  0: No helmet
  1: No vest
  2: Person 
  3: helmet 
  4: vest  
```

2. Now make a python file named train.py :

```http
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="config.yaml", epochs=5000)  # train the model
```

### Run the Training

In the terminal, run the following command:

```python train.py```

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 700 consecutive epoches.





## Testing for Verification

```pip install opencv-python```

```http
import cv2
from ultralytics import YOLO
img_pth = "1.jpeg"
model = YOLO("best.pt") 
results = model(source=img_pth)
res_plotted = results[0].plot()
cv2.imshow("result", res_plotted)
cv2.waitKey(0)
```

And finally I got my desired result.





## Conclusion

This documentary file summarizes the process of a Personal Protective Equipment (PPE) detection project, from data collection and labeling to code implementation and model evaluation. The project was conducted on a cloud machine with careful configuration, resulting in successful PPE detection using computer vision techniques. The optimized model achieved the desired accuracy with 700 echoes during training, ensuring workplace safety through automated PPE detection.