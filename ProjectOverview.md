# Project overview
A reliable object detection and classification is the basis for all self-driving vehicles. If we achieve this first step, wen can build more advanced algorithms for object tracking, prediction of object trajectories and also plan the trajectory of the ego vehicle on top.  
The overall goal of this project is to train a CNN based on a Single Shot Detector (SSD) Resnet 50 640x640 model. The model is trained with data from the Waymo Open Dataset, to detect vehicles, pedestrians and cyclists.  


# Set up

This section should contain a brief description of the steps to follow to run the code for this repository.


# Dataset

## Dataset analysis
We see that the number of objects per image have a great variance, i.e., in some images there are none and in others the image is densely spotted with objects, some of which are partially hidden. 

This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
## Cross validation

This section should detail the cross validation strategy and justify your approach.


# Training
## Reference experiment

This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

## Improve on the reference

This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.