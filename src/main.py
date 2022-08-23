#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 22 Aug, 2022

#Run our 3 experiments on all datasets

from cleanup import cleanup
from load_data_time_series.HAR.e4_wristband_Nov2019 import e4_load_dataset
from load_data_time_series.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
from load_data_time_series.HAR.UCI_HAR.uci_har_load_dataset import uci_har_load_dataset
from utils.sh_loader import sh_loco_load_dataset
from experiment1 import exp_1
from experiment2 import exp_2
from experiment3 import exp_3
import numpy as np
import pandas as pd

CLEANUP = False

#Dataset are returned in channels-last format
datasets = {
    'unimib' :  unimib_load_dataset,
    'twister' : e4_load_dataset,
    'uci har' : uci_har_load_dataset,
    'sussex huawei' : sh_loco_load_dataset
}

def channel_swap(X : np.ndarray) -> np.ndarray:
    """
    Return channels first array from channels last or vice versa
    """
    assert X.ndim == 3, "Data must be 3-dimensional to channel swap"
    return np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))

def run_and_write(exp: function, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, set: str, filename: str) -> None:
    """
    Open the resutlts file
    Run one experiment on one dataset
    Add the new results to the file
    """
    results = exp(X_train, y_train, X_test, y_test)
    results = pd.DataFrame.from_dict(results)
    results.to_csv(filename)


if __name__ == '__main__':
    """
    Run each experiment on each dataset, so... 12?
    """

    for set in datasets.keys():
        print (f"###   Running {set} ### ")
        
        ### Fetch Dataset ###
        X_train, y_train, X_test, y_test = datasets[set]()

        ### Channels first and flatten labels
        X_train = channel_swap(X_train)
        X_test = channel_swap(X_test)

        y_train = np.argmax(y_train, axis=-1)
        y_test = np.argmax(y_test, axis=-1)

        ### Run and Write
         


        if CLEANUP:
            cleanup()