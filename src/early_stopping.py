#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 20 Sep, 2022
#
#Make some nice models with a common interface

import numpy as np


class EarlyStopping_on_loss_minimum():
    """
    Set early_stop flag to True if tolerance cycles have passed without adeqate change
    """
    def __init__(self, tolerance=5, min_delta=0.01):

        self.patience = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.losses = list()

    def __call__(self, train_loss):
        self.losses.append(train_loss)
        if len(self.losses) == self.patience:
            if np.nanmax(self.losses) - train_loss < self.min_delta:
                self.counter += 1
            else:
                self.counter = 0
            if self.counter == self.patience:
                self.early_stop = True
            self.losses = self.losses[1:]