# Project overview
A reliable object detection and classification is the basis for all self-driving vehicles. If we achieve this first step, wen can build more advanced algorithms for object tracking, prediction of object trajectories and also plan the trajectory of the ego vehicle on top.  
The overall goal of this project is to train a CNN based on a Single Shot Detector (SSD) Resnet 50 640x640 model. The model is trained with data from the Waymo Open Dataset, to detect vehicles, pedestrians and cyclists.  


# Set up

This section should contain a brief description of the steps to follow to run the code for this repository.


# Dataset

## Dataset analysis
We have 97 .tfrecord files each containing 200 images.  
We see that the number of objects per image have a great variance, i.e., in some images there are none and in others the image is densely spotted (over 60) with objects, some of which are partially hidden.  
Furthermore I got the impression that most of the pictures where taken during daytime with a blue sky and there exist not many where the light or weather conditions are bad. 
As can be seen in the following graph, which is only based on the first 10000 images of each tfrecord file, most of the detected objects where cars, then pedestrians and only very few bycicles. Maybe we need to somehow balance this fact during training.  
![Compares occurrences of vehicles, pedestrians and cyclists in 10000 images](./Training_ClassOccurrenceComp.png)
One can see that the images are taken from multiple image sequences, because the same obejcts can be seen in several images.  
Is there an efficient way to randomly load images from the tfrecords?  
Since I found none, I just took the first 10000 images and plotted the class distribution for those. 


This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
## Cross validation
As I undersand it we should only move the complete files and do not split the .tfrecords.  

This section should detail the cross validation strategy and justify your approach.


# Training
## Reference experiment

This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

## Improve on the reference

This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.