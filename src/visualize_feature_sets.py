#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 13 Sep, 2022
#
#Visualize the Feature Spaces using UMAP

import os
from torch import Tensor
import numpy as np
from model_wrappers import Engineered_Features, Conv_AE, SimCLR_C, SimCLR_T, NNCLR_C, NNCLR_T, Supervised_C
from model_wrappers import SimCLR_R, NNCLR_R
import umap.umap_ as umap
from load_data_time_series.HAR.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from load_data_time_series.HAR.UniMiB_SHAR.unimib_shar_adl_load_dataset import unimib_load_dataset
from load_data_time_series.HAR.UCI_HAR.uci_har_load_dataset import uci_har_load_dataset
from utils.sh_loader import sh_loco_load_dataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

WRITE_FEATURES = False

umap_neighbors = 15
umap_dim = 3

class None_Extractor():
    def __init__(self,X, y): pass
    def fit(self, X, y, Xv, yv): pass
    def get_features(self, X):
        if X.ndim == 2:
            return X
        else:
            if X.shape[1] == 1: 
                return np.reshape(X, (X.shape[0], X.shape[2]))
            else:
                return np.array([np.linalg.norm(i, axis=0) for i in X])
        


feature_learners = {
    #"none" : None_Extractor,
    #"traditional" : Engineered_Features,
    "CAE" : Conv_AE,
    #"SimCLR + CNN" : SimCLR_C,
    #"SimCLR + T" : SimCLR_T,
    #"SimCLR + LSTM" : SimCLR_R,
    #"NNCLR + CNN" : NNCLR_C,
    #"NNCLR + T" : NNCLR_T,
    #"NNCLR + LSTM" : NNCLR_R,
    #"Supervised Convolutional" : Supervised_C
}

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

if __name__ == '__main__':

    if not os.path.exists('imgs'):
        os.mkdir('imgs')

    for set in datasets.keys():
        ### Fetch Dataset ###
        X_train, y_train, X_val, y_val, X_test, y_test = datasets[set](
            incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True
        )

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
        X_train = np.array([scaler.fit_transform(Xi) for Xi in X_train])
        X_val = np.array([scaler.fit_transform(Xi) for Xi in X_val])
        X_test = np.array([scaler.fit_transform(Xi) for Xi in X_test])

        for extractor_name in feature_learners.keys():
            #
            extractor = feature_learners[extractor_name](X_train, y_train)
            extractor.fit(X_train, y_train, X_val, y_val)

            features_train = extractor.get_features(X_train)
            #features_test = extractor.get_features(X_test)

            reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
            embedding = reducer.fit_transform(features_train)

            plt.figure()
            if umap_dim==2:
                plt.scatter(embedding[:,0], embedding[:,1], c=y_train)
            else:
                ax = plt.axes(projection ="3d")
                ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y_train)

            plt.savefig(f'imgs/{set}_with_{extractor_name}.pdf')
            del extractor
            del reducer
            del features_train





