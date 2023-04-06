#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Try out learned features with older tabular technique

import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_score
from model_wrappers import *
from early_stopping import EarlyStopping
from labelfix.src.labelfix import check_dataset

feature_learners = {
    "traditional" : Engineered_Features,
    #"CAE" : Conv_Autoencoder,
    "SimCLR + CNN" : SimCLR_C,
    "SimCLR + T" : SimCLR_T,
    "SimCLR + LSTM" : SimCLR_R,
    "NNCLR + CNN" : NNCLR_C,
    "NNCLR + T" : NNCLR_T,
    "NNCLR + LSTM" : NNCLR_R,
    "Supervised Convolutional" : Supervised_C
}

NUM_EPOCHS = 120
BATCH_SIZE = 32
NUM_WORKERS = 16

def setup_dataloader(X : np.ndarray, y : np.ndarray, shuffle: bool):
    torch_X = torch.Tensor(X)
    torch_y = torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
    dataloader = DataLoader(
        dataset=dataset, batch_size = BATCH_SIZE, shuffle=shuffle,
        drop_last=False, num_workers=NUM_WORKERS
    )
    return dataloader

def exp_4(
        X_train : np.ndarray,
        y_train : np.ndarray,
        X_val : np.ndarray,
        y_val : np.ndarray,
        X_test : np.ndarray,
        y_test : np.ndarray,
        set: str
) -> dict:

    results = {
        'set' : [],
        'features' : [],
        'noise percent' : [],
        '# instances' : [],
        'number mislabeled' : [],
        'prec of labels' : []
    }

    print ("Running Experiment 4  on ", set)
    print("Feature Extractors: ", ', '.join(feature_learners.keys()))

    if os.path.exists(f'temp/{set}_test_labels_high_noise.npy'):
        y_train_high = np.load(f'temp/{set}_train_labels_high_noise.npy')
        y_test_high = np.load(f'temp/{set}_test_labels_high_noise.npy')

        y_train_low = np.load(f'temp/{set}_train_labels_low_noise.npy')
        y_test_low = np.load(f'temp/{set}_test_labels_low_noise.npy')

    else:
        print("No label sets found. Please run experiments 1 and 2 first")
        return results
    
    for extractor in feature_learners.keys():

        if os.path.exists(f'temp/{set}_{extractor}_features_train_none_noise.npy'):
                f_train = np.load(f'temp/{set}_{extractor}_features_train_none_noise.npy', allow_pickle=True)
        else:
            print("No label sets found. Please run experiments 1 and 2 first")
            return results

        if os.path.exists(f'temp/{set}_{extractor}_features_test_none_noise.npy'):
            f_test = np.load(f'temp/{set}_{extractor}_features_test_none_noise.npy', allow_pickle=True)
        else:
            print("No label sets found. Please run experiments 1 and 2 first")
            return results
        
        noise_dic = {
            'low' : {
                'percent' : '5',
                'y_train' : y_train_low,
                'y_test_noisy' : y_test_low
            },
            'high' : {
                'percent' : '10',
                'y_train' : y_train_high,
                'y_test_noisy' : y_test_high
            },
        }

        for noise_level in noise_dic.keys():
            y_train_noisy = noise_dic[noise_level]['y_train']

            num_classes = np.nanmax(y_train)+1

            if os.path.exists(f'temp/{set}_{extractor}_features_train_none_noise.npy'):
                f_train = np.load(f'temp/{set}_{extractor}_features_train_none_noise.npy', allow_pickle=True)
            else:
                print("No label sets found. Please run experiments 1 and 2 first")
                return results
            
            labfix_results = check_dataset(f_train, y_train_noisy)
            sus_indices = labfix_results['indices']
            one_p = len(sus_indices) // 100
            prec_1p = np.count_nonzero(sus_indices[:one_p] in np.where(y_train_noisy != y_train)) / one_p
            prec_2p = np.count_nonzero(sus_indices[:one_p*2] in np.where(y_train_noisy != y_train)) / one_p*2
            prec_3p = np.count_nonzero(sus_indices[:one_p*3] in np.where(y_train_noisy != y_train)) / one_p*3

            results['set'].append(set)
            results['features'].append(extractor)
            results['noise percent'].append(noise_dic[noise_level]['percent'])
            results['# instances'].append(y_test.shape[0])
            results['number mislabeled'].append(np.count_nonzero(y_train != y_train_noisy))
            results['prec of labels at 1%'].append(prec_1p)
            results['prec of labels at 2%'].append(prec_2p)
            results['prec of labels at 3%'].append(prec_3p)
