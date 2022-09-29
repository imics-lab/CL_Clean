#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 20 Sep, 2022
#
#Make some nice models with a common interface

import numpy as np


class EarlyStopping():
    """
    Set early_stop flag to True if tolerance cycles have passed without adeqate change
    """
    def __init__(self, tolerance=5, min_delta=0.01, mode='minimum'):
        assert mode in ['minimum', 'maximum'], "Mode must be minimum or maximum"
        self.patience = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.losses = list()
        self.mode = mode

    def __call__(self, value):
        self.losses.append(value)
        if self.mode == 'minimum': 
            if len(self.losses) == self.patience:
                if np.nanmax(self.losses) - value < self.min_delta:
                    self.counter += 1
                else:
                    self.counter = 0
                self.losses = self.losses[1:]
        else:
            if len(self.losses) == self.patience:
                if value - np.nanmax(self.losses)  < self.min_delta:
                    self.counter += 1
                else:
                    self.counter = 0
                self.losses = self.losses[1:]

        if self.counter == self.patience:
                self.early_stop = True
        return
        