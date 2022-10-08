#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 08 Oct, 2022
#
#Make some nice models with a common interface

from main import load_synthetic_dataset, channel_swap
from model_wrappers import *
import umap.umap_ as umap
import matplotlib.pyplot as plt
from utils.add_nar import add_nar_from_array

umap_neighbors = 15
umap_dim = 2

feature_learners = {
    "None" : None_Extractor,
    "Engineered" : Engineered_Features,
    #"CAE" : Conv_AE,
    #"SimCLR + CNN" : SimCLR_C,
    #"SimCLR + Tran" : SimCLR_T,
    #"SimCLR + LSTM" : SimCLR_R,
    #"NNCLR + CNN" : NNCLR_C,
    #"NNCLR + Tran" : NNCLR_T,
    #"NNCLR + LSTM" : NNCLR_R,
    #"Sup. CNN" : Supervised_C
}

if __name__ == '__main__':
    X_train, y_train, X_val, y_val, X_test, y_test = load_synthetic_dataset(
        incl_xyz_accel=True, incl_rms_accel=False, incl_val_group=True
    )

    X_train = channel_swap(X_train)
    X_val = channel_swap(X_val)

    y_train_low, _, y_train_high, _ = add_nar_from_array(y_train, 2)

    for extractor_name in feature_learners.keys():
        extractor = feature_learners[extractor_name](X_train, y_train)
        extractor.fit(X_train, y_train, X_val, y_val)

        features_train = extractor.get_features(X_train)

        reducer = umap.UMAP(n_neighbors=umap_neighbors, n_components=umap_dim)
        embedding = reducer.fit_transform(features_train)

        #figure with no noise
        plt.figure()
        if umap_dim==2:
            plt.scatter(embedding[:,0], embedding[:,1], c=y_train)
        else:
            ax = plt.axes(projection ="3d")
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y_train)

        plt.savefig(f'imgs/syn_with_{extractor_name}_no_noise.pdf')

        #figure with 5% noise
        plt.figure()
        if umap_dim==2:
            plt.scatter(embedding[:,0], embedding[:,1], c=y_train)
        else:
            ax = plt.axes(projection ="3d")
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y_train_low)

        plt.savefig(f'imgs/syn_with_{extractor_name}_5per_noise.pdf')

        #figure with 10% noise
        plt.figure()
        if umap_dim==2:
            plt.scatter(embedding[:,0], embedding[:,1], c=y_train)
        else:
            ax = plt.axes(projection ="3d")
            ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=y_train_high)

        plt.savefig(f'imgs/syn_with_{extractor_name}_10per_noise.pdf')

        del extractor
        del reducer
        del features_train