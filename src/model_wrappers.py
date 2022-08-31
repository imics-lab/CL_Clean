#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

from pickletools import optimize
import os
from CL_HAR.models import backbones, frameworks, attention
import torch
import numpy as np
from torch import nn
from utils.ts_feature_toolkit import get_features_for_set
from torch.utils.data import DataLoader
import multiprocessing
from torchsummary import summary

EMBEDDING_WIDTH = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
#work around for mapping error
#torch.backends.cudnn.enabled = False
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


#Shameless theft: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopping():
    """
    Set early_stop flag to True if tolerance cycles have passed without adeqate change
    """
    def __init__(self, tolerance=5, min_delta=0.01):

        self.patience = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False
        self.losses = list()

    def __call__(self, train_loss):
        self.losses.append(train_loss)
        if len(self.losses) == self.patience:
            if np.max(self.losses) - train_loss > self.min_delta:
                self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
            self.losses = self.losses[1:]
            
            


class Engineered_Features():
    def __init__(self, X) -> None:
        pass
    def fit(self, X, y=None) -> None:
        """
        Just here to be here
        """
        pass

    def get_features(self, X) -> np.ndarray:
        return get_features_for_set(X)


class Conv_Autoencoder():
    def __init__(self, X, y) -> None:
        super(Conv_Autoencoder, self).__init__()
        self.model = backbones.CNN_AE(
            #channels first right?
            n_channels=X.shape[1],
            n_classes=np.max(y)+1,
            out_channels=EMBEDDING_WIDTH,
            backbone=True
        )
        self.model = self.model.to(device)
        self.criterion =  nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.patience = 7
        self.max_epochs = 200
        self.bath_size = 1
        summary(self.model, X.shape[1:])


    def fit(self, X, y=None) -> None:
        """
        Train cycle with early stopping
        """
        torch_X = torch.Tensor(X)
        torch_y = torch.Tensor(y)
        dataset = torch.utils.data.TensorDataset(torch_X, torch_y)
        dataloader = DataLoader(
            dataset=dataset, batch_size = self.bath_size, shuffle=False, drop_last=True
        )
        early_stopping = EarlyStopping(tolerance=self.patience, min_delta=0.1)
        self.model.to(device)
        for epoch in range(self.max_epochs):
            print("Epoch: ", epoch, end='.')
            total_loss = 0
            for x0, y0 in dataloader:
                x0 = x0.to(device)
                x_decoded, x_encoded = self.model(x0)
                #x_decoded comes back channels last
                #x_decoded = x_decoded.permute(0, 2, 1)
                # print(True in x0.detach() < 0)
                # print(True in x0.detach() > 1)
                loss = self.criterion(x0, x_decoded)
                total_loss += loss.detach()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            avg_loss = total_loss.detach()/len(dataloader)
            print('Train Loss: ', avg_loss.detach())
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print("Stopping early")
                



             

    def get_features(self, X) -> np.ndarray:
        pass


class SimCLR(nn.Module):
    def __init__(self, X, backbone='CNN') -> None:
        super(SimCLR, self).__init__()

    def fit(self, X, y=None) -> None:
        """
        Train cycle with early stopping
        """
        pass

    def get_features(self, X) -> np.ndarray:
        pass

class SimCLR_C(SimCLR):
    def __init__(self, X) -> None:
        super(SimCLR_C, self).__init__(X, 'CNN')

    def fit(self, X, y=None) -> None:
        pass

    def get_features(self, X) -> np.ndarray:
        pass

class SimCLR_T(SimCLR):
    def __init__(self, X) -> None:
        super(SimCLR_C, self).__init__(X, 'Transformer')

    def fit(self, X, y=None) -> None:
        pass

    def get_features(self, X) -> np.ndarray:
        pass

class NNCLR(nn.Module):
    def __init__(self, X, backbone='CNN') -> None:
        super(NNCLR, self).__init__()

    def fit(self, X, y=None) -> None:
        pass

    def get_features(self, X) -> np.ndarray:
        pass

class NNCLR_C(nn.Module):
    def __init__(self, X) -> None:
        super(NNCLR_C, self).__init__(X, 'CNN')

    def fit(self, X, y=None) -> None:
        pass

    def get_features(self, X) -> np.ndarray:
        pass

class NNCLR_T(nn.Module):
    def __init__(self, X) -> None:
        super(NNCLR_T, self).__init__(X, 'Transformer')

    def fit(self, X, y=None) -> None:
        pass

    def get_features(self, X) -> np.ndarray:
        pass

