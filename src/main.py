#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 22 Aug, 2022

#Run our 3 experiments on all datasets

from typing import Callable
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from cleanup import cleanup
from load_data_time_series.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from load_data_time_series.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
from load_data_time_series.HAR.UCI_HAR.uci_har_load_dataset import uci_har_load_dataset
from utils.sh_loader import sh_loco_load_dataset
from experiment1 import exp_1
from experiment2 import exp_2
from experiment3 import exp_3
import numpy as np
import pandas as pd
import os

CLEANUP = True

#Dataset are returned in channels-last format
datasets = {
    'unimib' :  unimib_load_dataset,
    #'twister' : e4_load_dataset,
    #'uci har' : uci_har_load_dataset,
    #'sussex huawei' : sh_loco_load_dataset
}

def channel_swap(X : np.ndarray) -> np.ndarray:
    """
    Return channels first array from channels last or vice versa
    """
    assert X.ndim == 3, "Data must be 3-dimensional to channel swap"
    return np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))

def run_and_write(
    exp: Callable, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val : np.ndarray,
    y_val : np.ndarray, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    set: str, 
    filename: str
) -> None:
    """
    Open the resutlts file
    Run one experiment on one dataset
    Add the new results to the file
    """
    results = exp(X_train, y_train, X_val, y_val, X_test, y_test, set)
    results = pd.DataFrame.from_dict(results)
    if not os.path.exists(filename):      
        results.to_csv(filename, index=False)
    else:
        file_frame = pd.read_csv(filename)
        file_frame = pd.concat([file_frame, results])
        file_frame.to_csv(filename, index=False)



if __name__ == '__main__':
    """
    Run each experiment on each dataset, so... 12?
    """
    NOW = datetime.now()

    for set in datasets.keys():
        print (f"###   Running {set} ### ")
        
        ### Fetch Dataset ###
        X_train, y_train, X_val, y_val, X_test, y_test = datasets[set](
            incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True
        )

        ### Sanity Checks ###
        print(f'Shape of train X: {X_train.shape}')
        print(f'Shape of val X: {X_val.shape}')
        print(f'Shape of test X: {X_test.shape}')

        ### Channels first and flatten labels
        ### Update: CL HAR prefers channels last
        X_train = channel_swap(X_train)
        X_test = channel_swap(X_test)
        X_val = channel_swap(X_val)

        y_train = np.array(np.argmax(y_train, axis=-1))
        y_val = np.array(np.argmax(y_val, axis=-1))
        y_test = np.array(np.argmax(y_test, axis=-1))


        ### Bound data to range -1 to 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        #scaler.fit(X_train)
        X_train = np.array([scaler.fit_transform(Xi) for Xi in X_train])
        X_val = np.array([scaler.fit_transform(Xi) for Xi in X_val])
        X_test = np.array([scaler.fit_transform(Xi) for Xi in X_test])

        ### Run and Write Experiments
        run_and_write(exp_1, X_train, y_train, X_val, y_val, X_test, y_test, set, "results/exp1_results_{}.csv".format(NOW))


        if CLEANUP:
            cleanup()