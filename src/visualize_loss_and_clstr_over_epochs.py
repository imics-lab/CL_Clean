#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 Oct, 2022
#
#visualize loss and clusterability

from main import load_synthetic_dataset, channel_swap
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from model_wrappers import Supervised_C

p = ['maroon', 'royalblue', 'forestgreen']

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic_dataset(
        incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True
    )

    X_train = channel_swap(X_train)
    X_val = channel_swap(X_val)

    y_train = np.argmax(y_train, axis=-1)
    y_val = np.argmax(y_val, axis=-1)

    extractor = Supervised_C(X_train, y_train)
    extractor.fit(X_train, y_train, X_val, y_val, record_values=True)