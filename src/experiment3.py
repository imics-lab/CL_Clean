#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 18 Aug, 2022
#
#Testing CL for Label Cleaning

#Experimental Design:
#   -add Noise at Random to one dataset
#   -extract features using 7 feature extractors: traditional, autoencoder, simclr+CNN, simclr+T, nnclr+CNN, nnclr+T
#   -train a down-stream classifier with features + cleaned or uncleaned labels
#   -measure the accuracy of the classifier on test labels

#Hypothesis:
#   Null: cleaning using CL will have no impact on downstream training
#   Alternative: cleaning using CL will positively impact downstream models

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from model_wrappers import *
from early_stopping import EarlyStopping

device = "cuda" if torch.cuda.is_available() else "cpu"

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

class Classifier(nn.Module):
    def __init__(self, n_classes) -> None:
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
           nn.LazyLinear(64),
           nn.Linear(64, 32),
           nn.Linear(32, n_classes)
        )
        self.model = self.model.to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def forward(self, X):
        return self.model(X)

def setup_dataloader(X : np.ndarray, y : np.ndarray, shuffle: bool):
    torch_X = torch.Tensor(X)
    torch_y = torch.Tensor(y)

    dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
    dataloader = DataLoader(
        dataset=dataset, batch_size = BATCH_SIZE, shuffle=shuffle,
        drop_last=False, num_workers=NUM_WORKERS
    )
    return dataloader


def exp_3(
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
        'accuracy on clean test' : [],
        'accuracy on noisy test' : []
    }

    print ("Running Experiment 3  on ", set)
    print("Feature Extractors: ", ', '.join(feature_learners.keys()))

    #Load label sets
    if os.path.exists(f'temp/{set}_test_labels_high_noise.npy'):
        y_train_high = np.load(f'temp/{set}_train_labels_high_noise.npy')
        y_test_high = np.load(f'temp/{set}_test_labels_high_noise.npy')

        y_train_low = np.load(f'temp/{set}_train_labels_low_noise.npy')
        y_test_low = np.load(f'temp/{set}_test_labels_low_noise.npy')

    else:
        print("No label sets found. Please run experiments 1 and 2 first")
        return results

    for extractor in feature_learners.keys():
        y_train_low_cleaned = np.load(f'temp/{set}_{extractor}_train_labels_low_noise_cleaned.npy')
        y_train_high_cleaned = np.load(f'temp/{set}_{extractor}_train_labels_high_noise_cleaned.npy')

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
            'none' : {
                'percent' : '0',
                'y_train' : y_train,
                'y_test_noisy' : y_test
            },
            'low' : {
                'percent' : '5',
                'y_train' : y_train_low,
                'y_test_noisy' : y_test_low
            },
            'low-cleaned' : {
                'percent' : '5-cleaned',
                'y_train' : y_train_low_cleaned,
                'y_test_noisy' : y_test_low
            },
            'high' : {
                'percent' : '10',
                'y_train' : y_train_high,
                'y_test_noisy' : y_test_high
            },
            'high-cleaned' : {
                'percent' : '10-cleaned',
                'y_train' : y_train_high_cleaned,
                'y_test_noisy' : y_test_high
            }
        }

        for noise_level in noise_dic.keys():
            y_train_noisy = noise_dic[noise_level]['y_train']
            y_test_noisy = noise_dic[noise_level]['y_test_noisy']

            num_classes = np.nanmax(y_train)+1

            model = Classifier(num_classes)

            train_loader = setup_dataloader(f_train, y_train_noisy, True)
            es = EarlyStopping(tolerance=7, min_delta=0.01, mode='minimum')

            #train our simple classifier
            for epoch in range(NUM_EPOCHS):
                print(f'Epoch {epoch}')
                total_loss = 0
                for i, (x0, y0) in enumerate(train_loader):
                    x0 = x0.to(device)
                    y0 = y0.type(torch.LongTensor).to(device)

                    model.optimizer.zero_grad()
                    if x0.size(0) != BATCH_SIZE:
                        continue

                    out = model(x0)
                    loss = model.criterion(out, y0)
                    loss.backward()
                    model.optimizer.step()

                    total_loss += loss.item()
                    if i%4 == 0: print('.', end='')
                
                total_loss /= BATCH_SIZE
                print('\n')
                es(total_loss)
                if es.early_stop:
                    print(f'Stopping early at epoch {epoch}')
                    break
                print('Train loss: ', total_loss)

            test_loader = setup_dataloader(f_test, y_test_noisy, False)
            #predict a label for every test instance
            y_pred = None
            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(device).float()
                    p = model(x)
                    if y_pred is None:
                        y_pred = p
                    else:
                        y_pred = torch.cat((y_pred, p))
            
            y_pred = y_pred.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=-1)


            results['set'].append(set)
            results['features'].append(extractor)
            results['noise percent'].append(noise_dic[noise_level]['percent'])
            results['# instances'].append(y_test.shape[0])
            results['number mislabeled'].append(np.count_nonzero(y_test != y_test_noisy))
            results['accuracy on clean test'].append(accuracy_score(y_test, y_pred))
            results['accuracy on noisy test'].append(accuracy_score(y_test_noisy, y_pred))

        #end for noise_level
    #end for extractor
    return results
#end exp3




            
    