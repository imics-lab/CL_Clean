#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise at Random to one dataset
#   -extract features using all feature extractors
#   -use HOC to clean label-set
#   -compute precision/recall of noisy label ID

#Hypothesis:
#   Null: there will be no trend between clusterability of feature space and precision of label cleaning
#   Alternate: more clusterable feature spaces will make more precise cleaning

import os
import numpy as np
from model_wrappers import Engineered_Features
from utils.add_nar import add_nar_from_array

feature_learners = {
    "traditional" : Engineered_Features,
    #"CAE" : Conv_Autoencoder,
    #"SimCLR + CNN" : SimCLR_C,
    #"SimCLR + T" : SimCLR_T,
    #"SimCLR + LSTM" : SimCLR_R,
    #"NNCLR + CNN" : NNCLR_C,
    #"NNCLR + T" : NNCLR_T,
    #"NNCLR + LSTM" : NNCLR_R,
    #"Supervised Convolutional" : Supervised_C
}

noise_levels = ['low', 'high']
percent_dic = {'low' : 5, 'high' : 10}

def exp_2(
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_val : np.ndarray,
        y_val : np.ndarray,
        X_test : np.ndarray,
        y_test : np.ndarray,
        set: str
) -> dict:
    results = {
        'set' : [],
        'features' : [],
        'noise percent' : [],
        'number mislabeled' : [],
        '# IDed as mislabeled' : [],
        'precision' : [],
        'recall' : [],
        'f1' : []
    }

    print ("Running Experiment 2  on ", set)
    print("Feature Extractors: ", ', '.join(feature_learners.keys()))

    #For all noise levels
    for noise_level in noise_levels:
        #Check for noisey labels, make them if necesary
        if os.path.exists(f'temp/{set}_test_labels_high_noise.npy'):
            y_train_high = np.load(f'temp/{set}_train_labels_high_noise.npy', dtype='int')
            y_val_high = np.load(f'temp/{set}_val_labels_high_noise.npy', dtype='int')
            y_test_high = np.load(f'temp/{set}_test_labels_high_noise.npy', dtype='int')

            y_train_low = np.load(f'temp/{set}_train_labels_low_noise.npy', dtype='int')
            y_val_low = np.load(f'temp/{set}_val_labels_low_noise.npy', dtype='int')
            y_test_low = np.load(f'temp/{set}_test_labels_low_noise.npy', dtype='int')
        else:
            num_classes = np.max(y_train)+1
            y_train_low, _, y_train_high, _ = add_nar_from_array(y_train, num_classes)
            y_val_low, _, y_val_high, _ = add_nar_from_array(y_val, num_classes)
            y_test_low, _, y_test_high, _ = add_nar_from_array(y_test, num_classes)

    noise_dic = {
        'none' : {
            'percent' : 0,
            'y_train' : y_train,
            'y_val' : y_val,
            'y_test' : y_test
        },
        'low' : {
            'percent' : 5,
            'y_train' : y_train_low,
            'y_val' : y_val_low,
            'y_test' : y_test_low
        },
        'high' : {
            'percent' : 10,
            'y_train' : y_train_high,
            'y_val' : y_val_high,
            'y_test' : y_test_high
        }
    }
        