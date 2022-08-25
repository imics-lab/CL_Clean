#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

from ..CL_HAR.models import backbones, attention, frameworks
import torch
import numpy as np
from torch import nn
from utils.ts_feature_toolkit import get_features_for_set

EMBEDDING_WIDTH = 64

class Engineered_Features():
    def __init__(self, X) -> None:
        pass
    def fit(self, X, y=None) -> None:
        pass
    def get_features(self, X) -> np.ndarray:
        return get_features_for_set(X)



class Conv_Autoencoder(nn.Module):
    def __init__(self, X) -> None:
        pass