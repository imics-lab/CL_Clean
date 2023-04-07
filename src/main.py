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
#from experiment3 import exp_3
from experiment4 import exp_4
from utils.gen_ts_data import generate_pattern_data_as_array
import numpy as np
import pandas as pd
import os
import torch
import argparse

CLEANUP = False

parser = argparse.ArgumentParser(description='argument setting of network')
parser.add_argument('--set', default='twister', type=str, help='Data Set')
parser.add_argument('--core', default='0', type=int, help='Cuda core')

def channel_swap(X : np.ndarray) -> np.ndarray:
    """
    Return channels first array from channels last or vice versa
    """
    assert X.ndim == 3, "Data must be 3-dimensional to channel swap"
    # return np.reshape(X, (X.shape[0], X.shape[2], X.shape[1]))
    return np.moveaxis(X, 2, 1)

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

def load_synthetic_dataset(incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True):
    NUM_TRAIN = 1001
    NUM_VAL = 101
    NUM_TEST = 101
    NUM_CLASSES = 2
    INSTANCE_LEN = 150

    params = {
        'avg_pattern_length' : [],
        'avg_amplitude' : [],
        'default_variance' : [],
        'variance_pattern_length' : [],
        'variance_amplitude' : []
    }

    for _ in range(NUM_CLASSES):
        params['avg_amplitude'].append(np.random.randint(0, 5))
        params['avg_pattern_length'].append(np.random.randint(5, 15))
        params['default_variance'].append(np.random.randint(1, 4))
        params['variance_pattern_length'].append(np.random.randint(5, 20))
        params['variance_amplitude'].append(np.random.randint(1, 5))

    train_set = np.zeros((NUM_TRAIN, INSTANCE_LEN))
    val_set = np.zeros((NUM_VAL, INSTANCE_LEN))
    test_set = np.zeros((NUM_TEST, INSTANCE_LEN))

    train_labels = []
    val_labels = []
    test_labels = []

    train_label_count = [0]*NUM_CLASSES
    val_label_count = [0]*NUM_CLASSES
    test_label_count = [0]*NUM_CLASSES

    for i in range (NUM_TRAIN):
        label = np.random.randint(0, NUM_CLASSES)
        # one_hot = np.zeros(NUM_CLASSES)
        # one_hot[label] = 1
        train_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        train_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        train_label_count[label] += 1

    for i in range (NUM_VAL):
        label = np.random.randint(0, NUM_CLASSES)
        val_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        val_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        val_label_count[label] += 1

    for i in range (NUM_VAL):
        label = np.random.randint(0, NUM_CLASSES)
        test_labels.append([0 if i!=label else 1 for i in range(NUM_CLASSES)])
        test_set[i, :] = generate_pattern_data_as_array(
            length=INSTANCE_LEN,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        test_label_count[label] += 1


    train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
    val_set = np.reshape(val_set, (val_set.shape[0], val_set.shape[1], 1))
    test_set = np.reshape(test_set, (test_set.shape[0], test_set.shape[1], 1))

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)

    print("Train labels: ", '\n'.join([str(i) for i in train_label_count]))
    print("Validatoions labels: ", '\n'.join([str(i) for i in val_label_count]))
    print("Test labels: ", '\n'.join([str(i) for i in test_label_count]))

    print("Train data shape: ", train_set.shape)
    print("Validation data shape: ", val_set.shape)
    print("Test data shape: ", test_set.shape)

    return train_set, train_labels, val_set, val_labels, test_set, test_labels



#Dataset are returned in channels-last format
datasets = {
    'synthetic' : load_synthetic_dataset,
    'unimib' :  unimib_load_dataset,
    'twister' : e4_load_dataset,
    'uci har' : uci_har_load_dataset,
    'sussex huawei' : sh_loco_load_dataset
}



if __name__ == '__main__':
    """
    Run each experiment on each dataset, so... 15?
    """
    torch.manual_seed(1899)
    np.random.seed(1899)

    NOW = datetime.now()

    args = parser.parse_args()

    #for set in datasets.keys():

    set = args.set
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
    #X_train = np.array([scaler.fit_transform(Xi) for Xi in X_train])
    #X_val = np.array([scaler.fit_transform(Xi) for Xi in X_val])
    #X_test = np.array([scaler.fit_transform(Xi) for Xi in X_test])

    ### Run and Write Experiments
    #run_and_write(exp_1, X_train, y_train, X_val, y_val, X_test, y_test, set, "results/exp1_results_{}_{}.csv".format(set, NOW))
    #run_and_write(exp_2, X_train, y_train, X_val, y_val, X_test, y_test, set, "results/exp2_results_{}_{}.csv".format(set, NOW))
    #run_and_write(exp_3, X_train, y_train, X_val, y_val, X_test, y_test, set, "results/exp3_results_{}_{}.csv".format(set, NOW))
    run_and_write(exp_4, X_train, y_train, X_val, y_val, X_test, y_test, set, "results/exp4_results_{}_{}.csv".format(set, NOW))


    if CLEANUP:
        cleanup()