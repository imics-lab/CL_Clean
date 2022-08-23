#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise Competely at Random to one dataset
#   -extract features using 7 feature extractors: traditional, autoencoder, simclr+CNN, simclr+T, nnclr+CNN, nnclr+T
#   -Fit neural models to the feature sets
#   -Calculate a true error rate for all 7 models

#Hypothesis:
#   Null: contrastive learning will have the same true error rate as AE and trad in the presence of NCAR
#   Alternative: models using CL will have lower TER

import torch
import numpy as np
from torch import Tensor

def exp_2(
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_test : np.ndarray,
    y_test : np.ndarray,
    set: str
):
    pass