#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

from operator import mod
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
from CL_HAR.utils import _logger
import fitlog

EMBEDDING_WIDTH = 64
SLIDING_WINDIW = 128
LR = 0.001
WEIGHT_DECAY = 0
NN_MEM = 1024

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
            
class ArgHolder():
    """
    Arguments for CL_HAR trainer
    """
    def __init__(self, 
        n_epoch : int, 
        batch_size : int, 
        framework : str,
        model_name : str,
        criterion : str,
        n_class : int
    ):
        self.n_epoch = n_epoch
        self.batch_size  = batch_size
        self.framework = framework
        self.model_name = model_name
        self.backbone = model_name
        self.cases = ""
        self.criterion = criterion
        self.weight_decay = 1e-5
        self.aug1 = "t_flip"
        self.aug2 = "noise"
        self.n_feature = EMBEDDING_WIDTH
        self.n_class = n_class
        self.len_sw = SLIDING_WINDIW
        self.lr=LR
        self.p = EMBEDDING_WIDTH
        self.phid = EMBEDDING_WIDTH
        self.weight_decay = WEIGHT_DECAY
        self.dataset = ""
        self.EMA = 0.996
        self.lambda1 = 1.0
        self.lambda2 = 1.0
        self.temp_unit = "gru"
        self.logdir = 'temp/'
        self.lr_cls = LR
        self.embedding_width = EMBEDDING_WIDTH
        self.mmb_size = NN_MEM

def setup_dataloader(X : np.ndarray, y : np.ndarray, args : ArgHolder):
    torch_X = torch.Tensor(X)
    torch_y = torch.Tensor(y)
    torch_d = torch.zeros(torch_y.shape)


    dataset = torch.utils.data.TensorDataset(torch_X, torch_y, torch_d)
    dataloader = DataLoader(
        dataset=dataset, batch_size = args.batch_size, shuffle=False, drop_last=False
    )
    return dataloader


class Engineered_Features():
    def __init__(self, X, y=None) -> None:
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
            n_classes=np.nanmax(y)+1,
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
    def __init__(self, X, y=None, backbone='CNN') -> None:
        super(SimCLR, self).__init__()
        assert backbone in ['CNN', 'Transformer'], 'Backbone type not supported now'

        self.args = ArgHolder(
            n_epoch=5,
            batch_size=32,
            framework="simclr",
            model_name='FCN' if backbone=='CNN' else 'Transformer',
            criterion="NTXent",
            n_class = np.nanmax(y)+1,
        )
        #Data is channels first
        self.args.len_sw = X.shape[2]
        self.args.n_feature = X.shape[1]

        model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = trainer.setup(self.args, device)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criterion = criterion
        self.logger = logger
        self.fitlog = fitlog
        self.classifier = classifier
        self.criterion_cls = criterion_cls
        self.optimizer_cls = optimizer_cls
        

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        """
        Train cycle with validation
        Runs through max number of epochs and then reloads best snapshot
        """
        
        train_dataloader = setup_dataloader(X_train, y_train, self.args)

        
        val_dataloader = setup_dataloader(X_val, y_val, self.args)

        best_model = trainer.train(
            train_loaders=[train_dataloader],
            val_loader=val_dataloader,
            model=self.model,
            logger=_logger('temp/simCLR_train_log.txt'),
            fitlog=self.fitlog,
            DEVICE=device,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            criterion=self.criterion,
            args=self.args
        )
        self.model.load_state_dict(best_model)
        return

    def __del__(self):
        trainer.delete_files(self.args)

class SimCLR_C(SimCLR):
    def __init__(self, X, y=None) -> None:
        super(SimCLR_C, self).__init__(X, y=y, backbone='CNN')
        


    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model.encoder(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return np.nanmax(fet.detach().cpu().numpy(), axis=2)
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)
        

class SimCLR_T(SimCLR):
    def __init__(self, X, y=None) -> None:
        super(SimCLR_T, self).__init__(X, y=y, backbone='Transformer')

    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model.encoder(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return np.nanmax(fet.detach().cpu().numpy(), axis=2)
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)

class NNCLR(nn.Module):
    def __init__(self, X, y=None, backbone='CNN') -> None:
        super(NNCLR, self).__init__()
        assert backbone in ['CNN', 'Transformer'], 'Backbone type not supported now'

        self.args = ArgHolder(
            n_epoch=5,
            batch_size=32,
            framework="nnclr",
            model_name='FCN' if backbone=='CNN' else 'Transformer',
            criterion="NTXent",
            n_class = np.nanmax(y)+1,
        )
        #Data is channels first
        self.args.len_sw = X.shape[2]
        self.args.n_feature = X.shape[1]

        model, optimizers, schedulers, criterion, logger, fitlog, classifier, criterion_cls, optimizer_cls = trainer.setup(self.args, device)
        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.criterion = criterion
        self.logger = logger
        self.fitlog = fitlog
        self.classifier = classifier
        self.criterion_cls = criterion_cls
        self.optimizer_cls = optimizer_cls

    def fit(self, X_train, y_train, X_val, y_val) -> None:
        """
        Train cycle with validation
        Runs through max number of epochs and then reloads best snapshot
        """
        train_dataloader = setup_dataloader(X_train, y_train, self.args)

        val_dataloader = setup_dataloader(X_val, y_val, self.args)

        best_model = trainer.train(
            train_loaders=[train_dataloader],
            val_loader=val_dataloader,
            model=self.model,
            logger=_logger('temp/simCLR_train_log.txt'),
            fitlog=self.fitlog,
            DEVICE=device,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            criterion=self.criterion,
            args=self.args
        )
        self.model.load_state_dict(best_model)
        return

    

class NNCLR_C(NNCLR):
    def __init__(self, X, y=None) -> None:
        super(NNCLR_C, self).__init__(X, y, 'CNN')

    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model.encoder(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return np.nanmax(fet.detach().cpu().numpy(), axis=2)
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)

class NNCLR_T(NNCLR):
    def __init__(self, X, y=None) -> None:
        super(NNCLR_T, self).__init__(X, y, 'Transformer')

    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model.encoder(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return np.nanmax(fet.detach().cpu().numpy(), axis=2)
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)

