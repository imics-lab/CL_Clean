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

import torch
import numpy as np
from torch import Tensor
from model_wrappers import Engineered_Features

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

def exp_2(
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_test : np.ndarray,
    y_test : np.ndarray,
    set: str
):
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

    #Check for noisey labels, make them if necesary