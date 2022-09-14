#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 08 Sep, 2022
#
#KNN-based noisy label detection based on:
#  https://arxiv.org/abs/2110.06283

from tkinter import N
import numpy as np
from utils import augmentation
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine

def compute_apparent_clusterability(
    fet : np.ndarray,
    y   : np.ndarray,
):
    """
    Compute that percentage of instances in the feature space that
    share an assigned label with their 2 nearest neighbors
    """
    neigh = NearestNeighbors(n_neighbors=2, radius=1.0, metric=cosine)
    neigh.fit(fet)
    clusterable_count = 0
    for i,f in enumerate(fet):
        #compute 3 nearest neighbors, discard 1st
        n = neigh.kneighbors(f, 3, return_distance=False)
        n = n[1:]
        if y[i] == y[n[0]] == y[n[1]]:
            clusterable_count += 1
    return clusterable_count/fet.shape[0]


def KNNLabel(
    fet: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    k : int
) -> np.ndarray:
    """
    Return an n x num_classes array of "fuzzy labels"
    for every instance in the feature set based on the
    labels of the k nearest neighbors
    """
    neigh = NearestNeighbors(n_neighbors=k+1, radius=1.0, metric=cosine)
    neigh.fit(fet)
    fuzzy_y = None
    for x0 in X:
        x0 = np.array([x0])
        nearest = neigh.kneighbors(x0, n_neighbors=None, return_distance=False)[1:]
        n_labels = np.array([np.count_nonzero(nearest==i) for i in range(num_classes)])
        n_labels = np.divide(n_labels, num_classes)
        if fuzzy_y is not None:
            fuzzy_y = np.concatenate((fuzzy_y, np.array([n_labels])), axis=0)
        else:
            fuzzy_y = np.array([n_labels])
    return fuzzy_y

def HOC(
    rounds : int,
    X : np.ndarray,
    y : np.ndarray
) -> np.ndarray:
    T_bar = None
    p_bar = None
    for _ in range(rounds):
        pass



def simiFeat(
    num_epochs: int,
    k: int,
    fet: np.ndarray,
    y: np.ndarray,
    method: str
) -> np.ndarray:
    """
    Return a cleaned label set using iterative KNN
    with fuzzy labeling
    """
    assert method in ["vote", "rank"], "Method must be vote or rank"
    if y.ndim > 1:
        y = np.argmax(y, axis=-1)
    y_clean = y.copy()
    num_classes = np.max(y)+1
    for n in range(num_epochs):
        y_prime = KNNLabel(fet, y_clean, num_classes, k)
        if method == "vote":
            y_clean = np.argmax(y_prime, axis=-1)
        else:
            pass


if __name__ == '__main__':
    X = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1],
        [1, 0, 0, 1, 1],
    ])

    y = np.array([0, 0, 0, 1, 1])
    compute_apparent_clusterability(X, y)
    #y_clean = simiFeat(10, 2, X, y, "vote")

