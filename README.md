# Object_Detection_and_Tracking_using_yolov8_and_Optimal_Tracker

## Environment setup

- Setup python environment
```
conda create -name optimal python=3.9
conda activate optimal
pip install -r requirements.txt
```

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

![val_batch0_pred](demo_images/val_batch1_pred.jpg)


## Result

- For real_time object tracking, evaluate `object_tracking.py`

![results](all-images/Screenshot(10))
![results](all-images/Screenshot(11))
![results](all-images/Screenshot(13))

```

