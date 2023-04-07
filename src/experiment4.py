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
from tensorflow import one_hot, make_ndarray

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
        'len of return' : [],
        'num mislabeled' : [],
        'prec of labels at 1%' : [],
        'prec of labels at 2%' : [],
        'prec of labels at 3%' : [],
        'prec of labels at 5%' : [],
        'prec of labels at 10%' : [],
        'rec of labels at 1%' : [],
        'rec of labels at 2%' : [],
        'rec of labels at 3%' : [],
        'rec of labels at 5%' : [],
        'rec of labels at 10%' : []
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
            
            #y_hot = np.eye(num_classes, dtype=int)[y_train_noisy] #cast to one-hot without tf
            labfix_results = check_dataset(f_train, y_train_noisy, 
                                           hyperparams={
                                            "input_dim": f_train.shape[1],
                                            "output_dim": num_classes,
                                            "num_hidden": 3,
                                            "size_hidden": 120,
                                            "dropout": 0.1,
                                            "epochs": 400,
                                            "learn_rate": 1e-2,
                                            "activation": "relu"
            })
            num_mislabeled = np.count_nonzero(y_train_noisy != y_train)
            mislabeled_indices = np.where(y_train_noisy != y_train)
            sus_indices = np.array(labfix_results['indices'])
            one_p = (len(sus_indices) // 100)
            np.save(f'temp/{set}_{extractor}_{noise_level}_sus_indices.npy', sus_indices)
            prec_1p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p]]) / one_p
            prec_2p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*2]]) / (one_p*2)
            prec_3p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*3]]) / (one_p*3)
            prec_5p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*5]]) / (one_p*5)
            prec_10p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*10]]) / (one_p*10)

            rec_1p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p]]) / len(mislabeled_indices)
            rec_2p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*2]]) / len(mislabeled_indices)
            rec_3p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*3]]) / len(mislabeled_indices)
            rec_5p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*5]]) / len(mislabeled_indices)
            rec_10p = np.count_nonzero([np.isin(i, mislabeled_indices) for i in sus_indices[:one_p*10]]) / len(mislabeled_indices)

            results['set'].append(set)
            results['features'].append(extractor)
            results['noise percent'].append(noise_dic[noise_level]['percent'])
            results['# instances'].append(y_test.shape[0])
            results['number mislabeled'].append(np.count_nonzero(y_train != y_train_noisy))
            results['len of return'].append(len(sus_indices))
            results['num mislabeled'].append(num_mislabeled)
            results['prec of labels at 1%'].append(prec_1p)
            results['prec of labels at 2%'].append(prec_2p)
            results['prec of labels at 3%'].append(prec_3p)
            results['prec of labels at 5%'].append(prec_5p)
            results['prec of labels at 10%'].append(prec_10p)
            results['rec of labels at 1%'].append(rec_1p)
            results['rec of labels at 2%'].append(rec_2p)
            results['rec of labels at 3%'].append(rec_3p)
            results['rec of labels at 5%'].append(rec_5p)
            results['rec of labels at 10%'].append(rec_10p)
    return results
