# -*- coding: utf-8 -*-
"""UCI_HAR_load_dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LSY4Cik5kPv3Y_XMhVWsepKM2cSrUBKp

#UCI_HAR_load_dataset.ipynb. 
Loads the [UCI HAR (Human Activity Recognition Using Smartphones)](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones#) dataset from the Internet repository and converts the acceleration data into numpy arrays while adhering to the general format of the [Keras MNIST load_data function](https://keras.io/api/datasets/mnist/#load_data-function).

This uses the original dataset which already has significant pre-processing such as separation into train/test and parsing into a 2.56 second sliding window with 50% overlap.   The acceleration imported is the body accel (total accel less 1g gravity) from train/Inertial Signals/body_acc_x_train.txt.  They gyro data is imported simliarly from body_gyro_train.txt.

So this function is more of a format converter from text to numpy.  It does not have the same level of adjustable parameters as some of the other loaders.

The train/test split is by subject as per the dataset documentation.

Example usage:

    x_train, y_train, x_test, y_test = uci_har_load_dataset()

Per the release notes use of this dataset in publications must be acknowledged by referencing the following publication   
> Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.   

Developed and tested using colab.research.google.com  
To save as .py version use File > Download .py

Author:  [Lee B. Hinkle](https://userweb.cs.txstate.edu/~lbh31/), [IMICS Lab](https://imics.wp.txstate.edu/), Texas State University, 2021

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

TODOs:
* Would be good to allocate specific subjects in validation vs stratify.
"""

import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
import requests #for downloading zip file
import time
from datetime import datetime, timedelta
import numpy as np
from tabulate import tabulate # for verbose tables, showing data
from tensorflow.keras.utils import to_categorical # for one-hot encoding
from sklearn.model_selection import train_test_split
from os import system

#for colab mount google drive and enter path to where the git repo was cloned
#my_path = '/content/drive/My Drive/Colab Notebooks/imics_lab_repositories/load_data_time_series_dev'
my_dir = '.'  # more flexible, enter full path if desired (not fully tested)
interactive = True # for working interactively in Jupyter notebook

# Don't run this cell if working interactively
interactive = False

# Build a top level (global) dictionary, each method can access and append
# to the contents and it is returned along with the ndarrays.
def start_dict():
    global info_dict
    info_dict = {
        'info_text':"Created by UCI_HAR_load_dataset on ",
        'sample_rate': 50,
        'labels':[],
        'channel_names': []
    }
    timestamp = time.strftime('%b-%d-%Y at %H:%M', time.localtime()) #UTC time
    info_dict['info_text'] += timestamp + '\n'
if (interactive):
    start_dict()
    for k, v in info_dict.items():
        print('{:<15s} {:<15s} {:<10s}'.format(k, str(v), str(type(v))))

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
def get_uci_har(chunk_size=128):
    """checks for existing copy, if not found downloads zip file from web archive
       saves and unzips into my_dir (global - defaults to '.')"""
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip'

    ffname = os.path.join(my_dir,'UCI_HAR_Dataset.zip')
    if (not os.path.isfile(ffname)):
        print("Downloading", ffname)
        r = requests.get(url, stream=True)
        with open(ffname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)
        info_dict['info_text'] += "\ndataset downloaded from " + url + '\n'
    else:
        print(ffname, "found, skipping download")
    if (not os.path.isdir(os.path.join(my_dir,'UCI HAR Dataset'))):
        print("Unzipping UCI_HAR_Dataset.zip file")
        shutil.unpack_archive(ffname,my_dir,'zip')
    else:
        print((os.path.join(my_dir,'UCI HAR Dataset')), 'directory found, skipping unzip')
if (interactive):
    get_uci_har()

def uci_har_load_dataset(
    verbose = True,
    drop_channels = [], # note this is zero indexed, check returned list
    sub_dict = {}, # for compatibility only, dataset already has assignments
    incl_val_group = False, # see note below. UCI HAR - already split train/test
    one_hot_encode = True,
    incl_xyz_accel=False, 
    incl_rms_accel=False
    ):
    """downloads and processes UCI HAR zip file into numpy arrays

    returns four or six numpy arrays and a dictionary with metadata
    x_train, y_train,<x_validate, y_validate>, x_test, y_test, info_dict"""
    print("Downloading and processing UCI HAR dataset")
    start_dict()
    get_uci_har()
    info_dict['info_text'] += 'UCI HAR is already split 70/30 by subject using a random allocation\n'
    #system('pwd')
    train_subs = np.loadtxt('UCI HAR Dataset/train/subject_train.txt', dtype = 'int8')
    test_subs = np.loadtxt('UCI HAR Dataset/test/subject_test.txt', dtype = 'int8')
    info_dict['info_text'] += "Training Subject Numbers" + str(np.unique(train_subs)) + '\n'
    info_dict['info_text'] += "Test Subject Numbers" + str(np.unique(test_subs)) + '\n'

    tmp_labels = np.genfromtxt('UCI HAR Dataset/activity_labels.txt',dtype='str')
    info_dict['labels'] = tmp_labels[:,1]
    # if (not os.path.isfile('./UCI_HAR_Dataset.zip')):
    #     print("Downloading UCI_HAR_Dataset.zip file")
    #     download_url('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip','./UCI_HAR_Dataset.zip')
    # if (not os.path.isdir('/content/UCI HAR Dataset')):
    #     print("Unzipping UCI_HAR_Dataset.zip file")
    #     shutil.unpack_archive('./UCI_HAR_Dataset.zip','.','zip')
    
    #Load .txt files as numpy ndarrays
    path_in = '/content/UCI HAR Dataset/'

    body_acc_x_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt')
    body_acc_y_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt')
    body_acc_z_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt')
    body_gyro_x_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt')
    body_gyro_y_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt')
    body_gyro_z_train = np.loadtxt('UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt')
    y_train = np.loadtxt('UCI HAR Dataset/train/y_train.txt')

    body_acc_x_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_acc_x_test.txt')
    body_acc_y_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_acc_y_test.txt')
    body_acc_z_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_acc_z_test.txt')
    body_gyro_x_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_gyro_x_test.txt')
    body_gyro_y_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_gyro_y_test.txt')
    body_gyro_z_test = np.loadtxt('UCI HAR Dataset/test/Inertial Signals/body_gyro_z_test.txt')
    y_test = np.loadtxt('UCI HAR Dataset/test/y_test.txt')

    # reshape to stack component accel and calculate magnitudes for accel and gyro
    acc_train = np.dstack((body_acc_x_train,body_acc_y_train,body_acc_z_train))
    gyro_train = np.dstack((body_gyro_x_train,body_gyro_y_train,body_gyro_z_train))
    acc_test = np.dstack((body_acc_x_test,body_acc_y_test,body_acc_z_test))
    gyro_test = np.dstack((body_gyro_x_test,body_gyro_y_test,body_gyro_z_test))
    ttl_acc_train = np.sqrt((acc_train[:,:,0]**2) + (acc_train[:,:,1]**2) + (acc_train[:,:,2]**2))
    ttl_acc_test = np.sqrt((acc_test[:,:,0]**2) + (acc_test[:,:,1]**2) + (acc_test[:,:,2]**2))
    ttl_gyro_train = np.sqrt((gyro_train[:,:,0]**2) + (gyro_train[:,:,1]**2) + (gyro_train[:,:,2]**2))
    ttl_gyro_test = np.sqrt((gyro_test[:,:,0]**2) + (gyro_test[:,:,1]**2) + (gyro_test[:,:,2]**2))
    x_train = np.dstack((acc_train,ttl_acc_train,gyro_train,ttl_gyro_train))
    x_test = np.dstack((acc_test,ttl_acc_test,gyro_test,ttl_gyro_test))

    #remove channels not needed - a bit brute force
    info_dict['channel_names'] = ['accel_x','accel_y','accel_z', 'accel_ttl',
                                  'gyro_x','gyro_y','gyro_z','gyro_ttl']
    # big credit to Ned Batchelder for pointing out reverse order deletion!
    # https://stackoverflow.com/questions/11303225/how-to-remove-multiple-indexes-from-a-list-at-the-same-time
    for index in sorted(drop_channels, reverse=True):
        del info_dict['channel_names'][index]
        x_train = np.delete(x_train, index, 2)
        x_test = np.delete(x_test, index, 2)
    if (one_hot_encode):
        y_train = y_train - 1 #original was 1 - 6
        y_test = y_test -1
        y_train = to_categorical(y_train, num_classes=6)
        y_test = to_categorical(y_test, num_classes=6)
        info_dict['info_text'] += 'y_test/train one-hot encoded\n'

    if (incl_val_group):
        print('\nWarning: UCI HAR is already split into train/test')
        print('The validation group is generated using sklearn stratify on train')
        print('It is not subject independent - confirm accuracy with test set')
        x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
        return x_train, y_train, x_validation, y_validation, x_test, y_test, info_dict
    else:
        return x_train, y_train, x_test, y_test #, info_dict

if __name__ == "__main__":
    x_train, y_train, x_test, y_test, info_dict = uci_har_load_dataset()
    print("\nUCI HAR returned arrays without validation group:")
    headers = ("Reshaped data","shape", "object type", "data type")
    mydata = [("x_train:", x_train.shape, type(x_train), x_train.dtype),
            ("y_train:", y_train.shape ,type(y_train), y_train.dtype),
            ("x_test:", x_test.shape, type(x_test), x_test.dtype),
            ("y_test:", y_test.shape ,type(y_test), y_test.dtype)]
    print(tabulate(mydata, headers=headers))
    print("\nContents of info_dict['info_text']")
    print(info_dict['info_text'])
    print("Contents of info_dict['channel_names']")
    print(info_dict['channel_names'])

    print('\n**** Rerunning this time with validation and dropping component accel/gyro')
    x_train, y_train, x_validation, y_validation, x_test, y_test, info_dict = uci_har_load_dataset(
        drop_channels = [0,1,2,4,5,6],
        incl_val_group=True)
    print("\nUCI HAR returned arrays with validation group:")
    headers = ("Reshaped data","shape", "object type", "data type")
    mydata = [("x_train:", x_train.shape, type(x_train), x_train.dtype),
            ("y_train:", y_train.shape ,type(y_train), y_train.dtype),
            ("x_validation:", x_validation.shape, type(x_validation), x_validation.dtype),
            ("y_validation:", y_validation.shape ,type(y_validation), y_validation.dtype),
            ("x_test:", x_test.shape, type(x_test), x_test.dtype),
            ("y_test:", y_test.shape ,type(y_test), y_test.dtype)]
    print(tabulate(mydata, headers=headers))
    print("\nContents of info_dict['info_text']")
    print(info_dict['info_text'])
    print("Contents of info_dict['channel_names']")
    print(info_dict['channel_names'])