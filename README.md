# Object_Detection_and_Tracking_using_yolov8_and_Optical_Tracker

## Environment setup

- Setup python environment
```
conda create -name optimal python=3.9
conda activate optimal
pip install -r requirements.txt
```

## Flowchart

![Flowchart](all-images/optical.png)

## Dataset labeling

The images for training has been labeled by labelimg. labelimg can be installed easily by
- Install labelimg
```
pip  install labelimg
labelimg
```
Image labeling contains two files i.e images and labels.

## Dataset Training

This code has been tested on  Python 3.9, Pytorch , CUDA 11.8

## Training Results

These are the reproduction results from the training.
- Confusion_Matrix
It tells that how many images are predicted correctly during training.
![confusion_matrix](all-images/confusion_matrix.png)

- Training_Results

![results](all-images/results.png)

Our labeled image 

![val_batch0_labels](all-images/val_batch1_labels.jpg)

Predicted image 

![val_batch0_pred](all-images/val_batch1_pred.jpg)


## Result

- For real_time object tracking without bounding box use 'opticalflow2.py`
![results](all-images/Screenshot(10).png)
![results](all-images/Screenshot(11).png)
![results](all-images/Screenshot(13).png)

  - For real_time object tracking with bounding box use 'opticalflow1.py`

![results](all-images/Screenshot(16).png)
![results](all-images/Screenshot(17).png)
![results](all-images/Screenshot(18).png)
![results](all-images/Screenshot(19).png)

```

