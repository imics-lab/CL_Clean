#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 08 Sep, 2022
#
#KNN-based noisy label detection based on:
#  https://arxiv.org/abs/2110.06283

import numpy as np
from src.utils import augmentation
from sklearn.neighbors import NearestNeighbors

def KNNLabel(
    fet: np.ndarray,
    y: np.ndarray,
    num_classes: int
    k : int
) -> np.ndarray:
    """
    Return an n x num_classes array of "fuzzy labels"
    for every instance in the feature set based on the
    labels of the k nearest neighbors
    """
    pass

def simiFeat(
    num_epochs: int,
    k: int,
    fet: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    for n in range(num_epochs):

        y_prime = None
