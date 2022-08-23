#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise at Random to one dataset
#   -extract features using 7 feature extractors: traditional, autoencoder, simclr+CNN, simclr+T, nnclr+CNN, nnclr+T
#   -predict a label using KNN for all instances of test data
#   -Compute:
#       P(mispredict | correct label)
#       P(mispredict | incorrect label)
#       P(predicted label == correct label)

#Hypothesis:
#   Null: contrastive learning will be no more likely to identify the correct label of data than AE or traditional
#   Alternative: labels predicted using CL will be more likely to match the true class

import torch
import os
from torch import Tensor
import numpy as np
from utils.add_nar import add_nar_from_array
from utils.ts_feature_toolkit import get_features_for_set, to_single_channel

feature_learners = {
    "traditional" : get_features_for_set
}

results = {
    'set' : [],
    'features' : [],
    'noise percent' : [],
    'P(mispred)' : [],
    'P(mispred|correct)' : [],
    'P(mispred|incorrect)' : [],
    'P(pred label = class)' : []
}


def exp_1(
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_test : np.ndarray,
        y_test : np.ndarray,
        set: str
    ) -> dict:
    """
    Run the experiment described on one train/test set.
    Return a dict with 7 rows of results
    """

    #Let's make some noise
    num_classes = np.max(y_train)+1
    X_train_low, _, X_train_high, _ = add_nar_from_array(X_train, num_classes)
    X_test_low, _, X_test_high, _ = add_nar_from_array(X_test, num_classes)

    #For each extractor with low noise labels
    for extractor in feature_learners.keys():
        print(f"## Experiment 1: Low Noise + {extractor} with {set}")
        if os.path.exists(f'temp/exp1_{set}_{extractor}_features_train_low_noise.npy'):
            f_train = np.load(f'temp/exp1_{set}_{extractor}_features_train_low_noise.npy')
        else:
            f_train = feature_learners[extractor](X_train, y_train)
            np.save(f_train, f'temp/exp1_{set}_{extractor}_features_train_low_noise.npy')





