# Project overview
A reliable object detection and classification is the basis for all self-driving vehicles. If we achieve this first step, wen can build more advanced algorithms for object tracking, prediction of object trajectories and also plan the trajectory of the ego vehicle on top.  
The overall goal of this project is to train a CNN based on a Single Shot Detector (SSD) Resnet 50 640x640 model. The model is trained with data from the [Waymo Open Dataset](https://waymo.com/open/) to detect vehicles, pedestrians and cyclists.  


# Set up
In my case the classroom workspace provided by Udacity was used.

## Prerequesites
In the classroom workspace the following packages were used:
* TensorFlow: Version 2.4.1
* Numpy: Version 1.18.5
* Matplotlib: Version 3.4.1

## Folder/File Structure:  
* After the split training data is stored under /home/workspace/data/waymo/
    - **train:** containinng the training data
    - **val:** containinng the validation data
    - **test:** containinng 3 files to test the model
* The experiments folder is organized as follow:
experiments/
    - **pretrained_model/** - contains the ssd_resnet50_v1_fpn_640x640_coco17_tpu-8 model which can be downloaded [here](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz)
    - **exporter_main_v2.py** - to create an inference model
    - **model_main_tf2.py** - to launch training
    - **reference/** - reference training with the unchanged config file
    - **experiment0/** - result of the first experiment
    - **experiment1/** - result of the second experiment
    - **experiment2/** - result of the third experiment
    - **label_map.pbtxt**
* Training_WeatherConditions.json: JSON file which assigns each tfrecord of the subfolder /home/workspace/data/waymo/training_and_validation to a selection of weather conditions. This file was created manually, as described in the Dataset Analysis section.  

For the github repo only the reference config and the config + animation of the last experiment is uploaded.
## Execution
1. Execute a splitting of the dataset, either by doing a random split with `split()` or using the function `weather_based_split()` both of which are located in the file create_split.py. Curently the default split is done with the function `weather_based_split()`.  
2. Run the training by executing the following commands:  
* **Training process:**
```
python experiments/model_main_tf2.py --model_dir=experiments/experiment3/ --pipeline_config_path=experiments/experiment3/pipeline_experiment3.config
```
* **Training monitoring with TensorBoard**
```
python -m tensorboard.main --logdir experiments/reference/
```
Due to GPU limitations it is not possible to run training and evaluation in parallel even with a batch size of 1. Therefore, the evaluation process is always launced after training is finished:
* **Evaluation process:**   
```
python experiments/model_main_tf2.py --model_dir=experiments/experiment3/ --pipeline_config_path=experiments/experiment3/pipeline_experiment3.config --checkpoint_dir=experiments/experiment3/
```
* **Model Export**
```
python experiments/exporter_main_v2.py --input_type image_tensor --pipeline_config_path experiments/experiment3/pipeline_experiment3.config --trained_checkpoint_dir experiments/experiment3/ --output_directory experiments/experiment3/exported/
```
* **Model Inference Animation**
```
python inference_video.py --labelmap_path label_map.pbtxt --model_path experiments/experiment3/exported/saved_model --tf_record_path /home/workspace/data/waymo/test/segment-12200383401366682847_2552_140_2572_140_with_camera_labels.tfrecord --config_path experiments/experiment3/pipeline_experiment3.config --output_path animation.gif
```

# Dataset
Before starting any training the [Waymo Open Dataset](https://waymo.com/open/) is inspected. 
## Dataset Analysis
In the Udacity workspace we have 97 .tfrecord files each containing 20 images. Through the `get_dataset()` call of `utils.py` we indefinitely read random elements from the tfrecords. The dataset covers different scenarios, from highways to urban environments and also varying lightning and weather conditions.  
In order to analyse the dataset we use the `Exploratory Data Analysis` notebook to display images at random and plot their corresponding bouding boxes and labels, as shown below:  
Example image night   |  Example during daytime
:---------------------------------------------:|:-------------------------:
![dataset analysis](./media/rand_example.png)  | ![dataset analysis](./media/rand_example2.png)  

Cars are denoted by red bounding boxes, pedestrians by blue ones and cyclists by green ones.  
Through the analysis we see that the number of objects per image have a great variance, i.e., in some images there are no objects and in others the image is densely spotted with vehicles, pedestrians and cyclists (over 60), some of which partially hidden. The following graph shows the average number of objects per image for each tfrecord. The analysis was done by iterating through all tfrecords and counting the objects of 10000 images for each record. The x-axis states the first 10 letters of each segment ID of the corresponding tfrecord. The main impression we get from the plot is that most of the detected objects where cars, then pedestrians and only very few cyclists. The code for the data acquisition and visualization can be found in the `Exploratory Data Analysis.ipynb` notebook.  
![Compares occurrences of vehicles, pedestrians and cyclists in 10000 images of each tfrecord file and normalizes the result](./media/Training_ClassOccurrenceComp.png)   
Furthermore most of the pictures where taken during daytime and there exist only few where the light or weather condition is bad. The following plot shows a classification of tfrecords to weather condition. As can be seen the largest portion can be assigned to a "good" weather, where the camera has a free view with good lightning conditions.  
![Shows the number of tfrecords distributed over weather conditions ](./media/Training_WeatherConditions.png)
The figure showing the weather distribution of the records was created by visualizing one image of each tfrecord file and manually classifying them in weather categories. Since each record contains an image sequence, it is acceptable to classify a record based on one image alone. The code for the visualization of one image and the plot can be found in the `Exploratory Data Analysis.ipynb` notebook.    
### Conclusion 
Maybe we need to somehow compensate for these two imbalances during training. The relatively small number of images with bad light conditions can be taken care of through data augmentations, but the fact that there are only very few images containing cyclists is harder to overcome.     

## Cross validation
As a last step before starting the training, we divide the data in three sets: 
* Training: Used to optimize the parameters of the model
* Validation: Used to evaluate the performance of the model during training, in order to adapt hyperparameters like the learning rate
* Testing: Dataset with which we evaluate the final models performance

Since Udacity already moved 3 records to the test folder we just needed to divide the 97 files contained in training_and_validation into training and validation sets.   
My appraoch was to split the files based on the weather condition. That is, I randomly choose a specified percentage from each weather condition and move the files to the train or validation folder. This should ensure that I am not training/validating the model with imbalanced data, for example nighttime images.    
For this project a split with 80% training and 20% validation data was chosen. 

# Training
## Reference experiment
The following plots show the performance of the reference model starting with predefined weights and config for 2500 optimization steps. The orange curve denotes the training process and the blue dots correspond to the evaluation after training:  
![Reference run results](./media/ReferenceRun_TensorBoard.png)
Due to the limited GPU power it was only possible to run a evaluation or a training process, but not both in parallel, even when the batch size is reduced to 1. Therefore, above plots only show the evaluation for the model after training, that is, one blue dot after 2500 optimization steps.  
From this initial run we see the following two points:
* The learning rate in the beginning is too large since the loss function has some sudden increases during the first 900 steps
* The model is overfitting a little bit since the evaluation loss is approximately 0.5 points above the trainig loss
These two shortcomings are addressed in the next section. 
## Improve on the reference
Starting from the reference experiment some modifications were tested in order to improve on the model performance. Since our dataset is small the first steps are to augment the data by the following methods:
* Random grayscale conversion with probability of 0.1
* Random brightness adjustment with max_delta 0.1
* Random contrast adjustment with values between 0.6 and 1.0
* Random cropping based on the ssd_random_crop 

Through this the small dataset can be extended and the model is for example confronted with grayscale images as shown below:
![Grayscale image augmentation](./media/Aug_GrayScale.png)

Furthermore, the learning rate is reduced and defined by the following parameters:
* learning_rate_base: 0.0004
* total_steps: 2500
* warmup_learning_rate: 0.0001
* warmup_steps: 200

Another parameter that was adapted is the batch size, which was doubled to 4. The reason is that during experimenting I encountered many jumps in the loss curve and as described in the last section this can be reduced by the batch size value.  
As a result the following performance was achieved:  
Loss:
![Grayscale image augmentation](./media/Improved_Loss.png)
Precision:
![Grayscale image augmentation](./media/Improved_DetectionBox_Precision.png)
Recall:
![Grayscale image augmentation](./media/Improved_DetectionBox_Recall.png)
As with the reference model it was not possible to run evaluation and training in parallel, therefore only a single evaluation run was executed after the training process.  
By taking the steps described above we decreased the total loss and also managed to reduce the overfitting. But in my opinion one problem still remains: The imbalance of the dataset. On the one hand the images contain only very few cyclists and on the other hand I believe the images taken during bad lightning or bad weather conditions are too few. Wthin limits the last point can be reduced by extensive augmentations, but they do not provide the solution for the lack of cyclists.  

### Fluctuation
For the fluctuation of the training loss which I encountered during the experiments I have two possible explanations:  
* By using a very small batch_size of 1 or 2, we are doing gradient steps based on small input data. Thus, the probability of an update based on unbalanced input is larger and can lead to an increase of the global loss, i.e., we take a step away from the local minimum. When the batch_size is larger, such effects should have a smaller impact, since an update is based on more and hopefully more balanced data.
* Another explanation may be the learning rate which can be chosen too large and therefore we are jumping around the local minimum but not able to get significantly closer. This is the reason why I also chose to use a smaller learning rate.
