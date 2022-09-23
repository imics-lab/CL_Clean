#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise at Random to one dataset
#   -extract features using all feature extractors
#   -use HOC to clean label-set
#   -compute precision/recall of noisy label ID

#Hypothesis:
#   Null: there will be no trend between clusterability of feature space and precision of label cleaning
#   Alternate: more clusterable feature spaces will make more precise cleaning

WRITE_LABELS = True

import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from model_wrappers import Engineered_Features
from utils.add_nar import add_nar_from_array
from cleaner import simiFeat

feature_learners = {
    "traditional" : Engineered_Features,
    #"CAE" : Conv_Autoencoder,
    #"SimCLR + CNN" : SimCLR_C,
    #"SimCLR + T" : SimCLR_T,
    #"SimCLR + LSTM" : SimCLR_R,
    #"NNCLR + CNN" : NNCLR_C,
    #"NNCLR + T" : NNCLR_T,
    #"NNCLR + LSTM" : NNCLR_R,
    #"Supervised Convolutional" : Supervised_C
}

def exp_2(
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
        '# IDed as mislabeled' : [],
        '# correct after clean' : [],
        'precision' : [],
        'recall' : [],
        'f1' : []
    }

    print ("Running Experiment 2  on ", set)
    print("Feature Extractors: ", ', '.join(feature_learners.keys()))

    #Check for noisey labels, make them if necesary
    if os.path.exists(f'temp/{set}_test_labels_high_noise.npy'):
        y_train_high = np.load(f'temp/{set}_train_labels_high_noise.npy')
        y_val_high = np.load(f'temp/{set}_val_labels_high_noise.npy')
        y_test_high = np.load(f'temp/{set}_test_labels_high_noise.npy')

        y_train_low = np.load(f'temp/{set}_train_labels_low_noise.npy')
        y_val_low = np.load(f'temp/{set}_val_labels_low_noise.npy')
        y_test_low = np.load(f'temp/{set}_test_labels_low_noise.npy')
    else:
        num_classes = np.max(y_train)+1
        y_train_low, _, y_train_high, _ = add_nar_from_array(y_train, num_classes)
        y_val_low, _, y_val_high, _ = add_nar_from_array(y_val, num_classes)
        y_test_low, _, y_test_high, _ = add_nar_from_array(y_test, num_classes)

    noise_dic = {
        'none' : {
            'percent' : 0,
            'y_train' : y_train,
            'y_val' : y_val,
            'y_test' : y_test
        },
        'low' : {
            'percent' : 5,
            'y_train' : y_train_low,
            'y_val' : y_val_low,
            'y_test' : y_test_low
        },
        'high' : {
            'percent' : 10,
            'y_train' : y_train_high,
            'y_val' : y_val_high,
            'y_test' : y_test_high
        }
    }

    #For all feature extractors and all noise levels
    for extractor in feature_learners.keys():
        for noise_level in noise_dic.keys():

            y_train_noisy = noise_dic[noise_level]['y_train']
            y_val_noisy = noise_dic[noise_level]['y_val']
            y_test_noisy = noise_dic[noise_level]['y_test']

            #Check for numpy file of feature
            if os.path.exists(f'temp/{set}_{extractor}_features_train_{noise_level}_noise.npy'):
                f_train = np.load(f'temp/{set}_{extractor}_features_train_{noise_level}_noise.npy', allow_pickle=True)
            else:
                f_learner = feature_learners[extractor](X_train, y=y_train_noisy)
                f_learner.fit(X_train, y_train_noisy, X_val, y_val_noisy)
                f_train = f_learner.get_features(X_train)

            if os.path.exists(f'temp/{set}_{extractor}_features_test_{noise_level}_noise.npy'):
                f_test = np.load(f'temp/{set}_{extractor}_features_test_{noise_level}_noise.npy', allow_pickle=True)
            else:
                f_test = f_learner.get_features(X_test)

            y_train_cleaned = simiFeat(10, 3, f_train, y_train_noisy, "rank")
            y_test_cleaned = simiFeat(10, 3, f_test, y_test_noisy, "rank")

            if WRITE_LABELS:
                with open(f'temp/{set}_{extractor}_train_labels_{noise_level}_noise_cleaned.npy', 'wb+') as f:
                    np.save(f, y_train_cleaned)
                with open(f'temp/{set}_{extractor}_test_labels_{noise_level}_noise_cleaned.npy', 'wb+') as f:
                    np.save(f, y_test_cleaned)

            results['set'].append(set)
            results['features'].append(extractor)
            results['noise percent'].append(noise_dic[noise_level]['percent'])
            results['# instances'].append(y_test.shape[0])
            results['number mislabeled'].append(np.count_nonzero(y_test != y_test_noisy))
            results['# IDed as mislabeled'].append(np.count_nonzero(y_test_cleaned != y_test_noisy))
            results['# correct after clean'].append(np.count_nonzero(y_test_cleaned == y_test))
            results['precision'].append(precision_score(y_test, y_test_cleaned, average='micro'))
            results['recall'].append(recall_score(y_test, y_test_cleaned, average='micro'))
            results['f1'].append(f1_score(y_test, y_test_cleaned, average='micro'))





        