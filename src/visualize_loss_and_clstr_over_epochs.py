#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 Oct, 2022
#
#visualize loss and clusterability

from main import channel_swap
from load_data_time_series.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from model_wrappers import Supervised_C
import pandas as pd

p = ['maroon', 'royalblue', 'forestgreen']

if __name__ == '__main__':
    # X_train, y_train, X_val, y_val, X_test, y_test = e4_load_dataset(
    #     incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True
    # )

    # X_train = channel_swap(X_train)
    # X_val = channel_swap(X_val)

    # y_train = np.argmax(y_train, axis=-1)
    # y_val = np.argmax(y_val, axis=-1)

    # extractor = Supervised_C(X_train, y_train)
    # extractor.fit(X_train, y_train, X_val, y_val, record_values=True)

    train_info = pd.read_csv('results/train_values.csv')

    fig,ax = plt.subplots()
    ax.plot(range(120), train_info['Train Loss'], c=p[0])
    ax.plot(range(120), train_info['Val Loss'], c=p[1])
    ax.set_xlabel('Training Epochs')
    ax.set_ylabel('Loss')
    ax2 = ax.twinx()
    ax2.set_ylabel('Clusterability')
    ax2.plot(range(120), train_info['Clusterability'], c=p[2])
    plt.title('Loss and Clusterability during Trainging', fontsize=18)
    fig.savefig('imgs/loss_and_clstr_during_train.pdf')