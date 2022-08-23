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
from torch import Tensor
import numpy as np
from src.utils.add_nar import add_nar

def exp_1(
    X_train : np.ndarray,
    y_train : np.ndarray,
    X_test : np.ndarray,
    y_test : np.ndarray,
    set: str
):
    pass