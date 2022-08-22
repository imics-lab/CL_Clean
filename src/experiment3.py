#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise Competely at Random to one dataset
#   -extract features using 7 feature extractors: traditional, autoencoder, simclr+CNN, simclr+T, nnclr+CNN, nnclr+T
#   -predict using KNN following the technique of Experiment 1
#   -follow a cycle of relabeling followed by repredicting until stasis
#   -train and fit neural models to the raw data using the cleaned train labels
#   -measure accuracy and f1 of fitted models on test data
#   -proceed through k folds of train test splits

#Hypothesis:
#   Null: cleaning using CL will have no impact on downstream training
#   Alternative: cleaning using CL will positively impact downstream models

import torch
from torch import Tensor

def exp_3(
    X : Tensor,
    y : Tensor,
):
    pass