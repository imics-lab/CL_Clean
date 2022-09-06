#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

from pickletools import optimize
import os
from CL_HAR.models import backbones, frameworks, attention
from CL_HAR import trainer
import torch
import numpy as np
from torch import nn
from utils.ts_feature_toolkit import get_features_for_set
from torch.utils.data import DataLoader
import multiprocessing
from torchsummary import summary
from CL_HAR.utils import logger
import fitlog

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
    def fit(self, X_train, y_train=None, X_val=None, y_val=None) -> None:
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
        #This started as a typo, but I like it
        #Bath == Batch
        self.bath_size = 32
        summary(self.model, X.shape[1:])


    def fit(self, X_train, y_train=None, X_val=None, y_val=None) -> None:
        """
        Train cycle with early stopping
        """
        train_torch_X = torch.Tensor(X_train)
        train_torch_y = torch.Tensor(y_train)
        train_dataset = torch.utils.data.TensorDataset(train_torch_X, train_torch_y)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size = self.bath_size, shuffle=False, drop_last=True
        )
        early_stopping = EarlyStopping(tolerance=self.patience, min_delta=0.1)
        self.model.to(device)
        for epoch in range(self.max_epochs):
            print("Epoch: ", epoch, end='.')
            total_loss = 0
            for x0, y0 in train_dataloader:
                x0 = x0.to(device)
                x_decoded, x_encoded = self.model(x0)
                #x_decoded comes back channels last
                #x_decoded = x_decoded.permute(0, 2, 1)
                #print(True in x0.detach() < 0)
                #print(True in x0.detach() > 1)
                loss = self.criterion(x0, x_decoded)
                total_loss += loss.detach()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            avg_loss = total_loss.detach()/len(train_dataloader)
            print('Train Loss: ', avg_loss.detach())
            early_stopping(avg_loss)
            if early_stopping.early_stop:
                print("Stopping early")
    

    def get_features(self, X) -> np.ndarray:
        if device == 'cuda':
            return self.model(X).cpu().detach().numpy()
        else:
            return self.model(X).detach().numpy()


class SimCLR(nn.Module):
    def __init__(self, X, backbone='CNN') -> None:
        super(SimCLR, self).__init__()
        if backbone=='CNN':
            self.model = frameworks.SimCLR(backbone='FCN')
        elif backbone == 'Transformer':
            self.model = frameworks.SimCLR(backbone='Transformer')

        self.model = self.model.to(device)
        self.criterion =  nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.patience = 7
        self.max_epochs = 200
        self.bath_size = 32
        summary(self.model, X.shape[1:])

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        """
        Train cycle with early stopping
        """
        train_torch_X = torch.Tensor(X_train)
        train_torch_y = torch.Tensor(y_train)
        val_torch_X = torch.Tensor(X_val)
        val_torch_y = torch.Tensor(y_val)

        train_dataset = torch.utils.data.TensorDataset(train_torch_X, train_torch_y)
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size = self.bath_size, shuffle=False, drop_last=True
        )

        val_dataset = torch.utils.data.TensorDataset(val_torch_X, val_torch_y)
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size = self.bath_size, shuffle=False, drop_last=True
        )

        self.model = trainer.train(
            train_loaders=[train_dataloader],
            val_loader=[val_dataloader],
            model=self.model,
            logger=logger('temp/simCLR_train_log.txt'),
            fitlog=fitlog,
            DEVICE=device,
            optimizers=self.optimizer,
            schedulers=None,
            criterion=self.criterion
        )


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

