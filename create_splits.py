import argparse
import glob
import os
import random
import json

import numpy as np

from utils import get_module_logger


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    
    val_portion = 0.20    # ->  80% is used for training 
    train_des = os.path.join('/home/workspace/data/waymo','train')
    val_des = os.path.join('/home/workspace/data/waymo','val')
    # Check if destination dirs exist and if not create them
    if not os.path.isdir(train_des):
        os.makedirs(train_des)
    if not os.path.isdir(val_des):
        os.makedirs(val_des)

    # Shuffle files and split them according to val_portion in training and validation
    files = glob.glob(os.path.join(data_dir,'*.tfrecord'))
    np.random.shuffle(files)
    train_files, val_files = np.split(np.array(files), [int(len(files)* (1 - val_portion))])

    for record in train_files:
        filename = os.path.basename(record)
        os.rename(record, os.path.join(train_des,filename))
    for record in val_files:
        filename = os.path.basename(record)
        os.rename(record, os.path.join(val_des,filename))


def weather_based_split(source, destination, weather_dict):
    """
    The split is executed based on a json file Training_WeatherConditions.json. This file
    contains all 97 .tfrecords which are assigned to specified weather conditions.
    This split tries to create a balanced split, so that each set (training, validation)
    has the same ratio of rainy, sunny, cloudy images.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
        - weather_dict[dict]: dictionary containing the assignment of tfrecords and weather condition
    """
    val_portion = 0.20    # ->  80% is used for training 
    train_des = os.path.join(destination,'train')
    val_des = os.path.join(destination,'val')
    # Check if destination dirs exist
    if not os.path.isdir(train_des):
        os.makedirs(train_des)
    if not os.path.isdir(val_des):
        os.makedirs(val_des)

    train_files = []
    val_files = []
    # Read filenames from dictionary, shuffle files with similar weather conditions and then do the splitting
    for weather in weather_dict: 
        files = weather_dict[weather]
        # Shuffle files and split them according to val_portion in training and validation
        np.random.shuffle(files)
        train_tmp, val_tmp = np.split(np.array(files), [int(len(files)* (1 - val_portion))])
        train_files.extend(train_tmp)
        val_files.extend(val_tmp)

    # Execute the split by moving the files to new location
    distribution = {"train": [], "val":[]}
    for record in train_files:
        dst_filename = os.path.basename(record)
        os.rename(os.path.join(source,record), os.path.join(train_des,dst_filename))    
        distribution["train"].append(dst_filename)
    for record in val_files:
        dst_filename = os.path.basename(record)
        os.rename(os.path.join(source,record), os.path.join(val_des,dst_filename))
        distribution["val"].append(dst_filename)

    # Write file distribution to file for later analysis
    with open('Train_Val_Dist.json', 'w') as f:
        print(distribution, file=f)
        

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    
    np.random.seed(5)                       # Reproducable shuffling in split functions  
    #split(args.data_dir)
    
    # Load dictionary containing the weather condition - tfrecord assignment
    with open('Training_WeatherCondition.json') as f:
        Training_WeatherCondition = json.load(f)

    weather_file_dict = Training_WeatherCondition['Classification']
    weather_based_split(args.data_dir, '/home/workspace/data/waymo', weather_file_dict)
    
    