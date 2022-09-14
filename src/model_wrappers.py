#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

#####################################################################
#Supported Feature Learners:
#  Engineered_Features -> Engineerd using signal processing
#  SimCLR_C     -> SimCLR w/ CNN encoder
#  SimCLR_T     -> SimCLR w/ Transformer encoder
#  SimCLR_R     -> SimCLR w/ Convolutional LSTM encoder
#  NNCLR_C      -> NNCLR w/ CNN encoder
#  NNCLR_T      -> NNCLR w/ Transformer encoder
#  NNCLR_R      -> NNCLR w/ Covolutional LSTM encoder
#  Supervised_C -> Supervised learning w/ CNN encoder
#  Conv_AE      -> Convolutional Autoencoder
#####################################################################

from multiprocessing import reduction
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


EMBEDDING_WIDTH = 96
SLIDING_WINDIW = 128
LR = 0.001
WEIGHT_DECAY = 1e-5
NN_MEM = 1024 #size in megabytes
CL_EPOCHS = 120

LOG = _logger('temp/train_log.txt')

device = "cuda" if torch.cuda.is_available() else "cpu"
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
            if np.nanmax(self.losses) - train_loss < self.min_delta:
                self.counter += 1
            else:
                self.counter = 0
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
        self.model_name = framework + ' ' + model_name
        self.backbone = model_name
        self.cases = ""
        self.criterion = criterion
        self.weight_decay = 1e-5
        self.aug1 = "noise"
        self.aug2 = "negate"
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


class Conv_AE(nn.Module):
    def __init__(self, X, y) -> None:
        super(Conv_AE, self).__init__()
        self.model = backbones.CNN_AE(
            #channels first right?
            n_channels=X.shape[1],
            n_classes=np.nanmax(y)+1,
            out_channels=EMBEDDING_WIDTH,
            backbone=False
        )
        self.model = self.model.to(device)
        
        self.args = ArgHolder(
            n_epoch=100,
            batch_size=32,
            framework="",
            model_name="",
            criterion='mse',
            n_class=np.nanmax(y)+1,
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())


    def fit(self, X_train, y_train=None, X_val=None, y_val=None) -> None:
        """
        Train cycle with early stopping
        """
        train_loader = setup_dataloader(X_train, y_train)
        val_loader = setup_dataloader(X_val, y_val)
        es = EarlyStopping(tolerance=10, min_delta=0.001)
        for epoch in range(self.args.n_epoch):
            print(f'Epoch {epoch}:')
            total_loss = 0
            val_loss = 0
            for i, (x0, y0, d) in enumerate(train_loader):
                self.optimizer.zero_grad()
                x0 = x0.to(device)
                y0 = y0.type(torch.LongTensor)
                y0 = y0.to(device)
                if x0.size(0) != self.args.batch_size:
                    continue
                out, f = self.model(x0)
                loss = self.criterion(out, x0)
                total_loss += loss.item()
                total_loss /= self.args.batch_size
                loss.backward()
                self.optimizer.step()
                if i%4 == 0: print('.', end='')
            print('\n')
            with torch.no_grad():
                for (x1, y1, d) in val_loader:
                    x1 = x1.to(device)
                    y1 = y1.type(torch.LongTensor)
                    y1 = y1.to(device)
                    out, f = self.model(x1)
                    loss = self.criterion(out, x1)
                    val_loss += loss.item()
                    val_loss /= self.args.batch_size
            es(val_loss)
            if es.early_stop:
                print(f'Stopping early at epoch {epoch}')
                return

            print('Train loss: ', total_loss)
            print('Validation loss: ', val_loss)

    

    def get_features(self, X) -> np.ndarray:
        if device == 'cuda':
            return self.model(X).cpu().detach().numpy()
        else:
            return self.model(X).detach().numpy()


class SimCLR(nn.Module):
    def __init__(self, X, y=None, backbone='CNN') -> None:
        super(SimCLR, self).__init__()
        assert backbone in ['CNN', 'Transformer', 'DeepConvLSTM'], 'Backbone type not supported now'

        self.args = ArgHolder(
            n_epoch=CL_EPOCHS,
            batch_size=32,
            framework="simclr",
            model_name='FCN' if backbone=='CNN' else backbone,
            criterion="NTXent",
            n_class = np.nanmax(y)+1,
        )
        if self.args.backbone == "DeepConvLSTM" : self.args.backbone = 'DCL'

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
            logger=LOG,
            fitlog=self.fitlog,
            DEVICE=device,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            criterion=self.criterion,
            args=self.args
        )
        self.model.load_state_dict(best_model)
        self.model = trainer.lock_backbone(self.model, self.args)
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
                _, f = self.model(x)
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
                _, f = self.model(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return fet.detach().cpu().numpy()
        else:
            return fet.detach().numpy()

class SimCLR_R(SimCLR):
    def __init__(self, X, y=None) -> None:
        super(SimCLR_R, self).__init__(X, y=y, backbone='DeepConvLSTM')
        


    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return fet.detach().cpu().numpy()
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)

class NNCLR(nn.Module):
    def __init__(self, X, y=None, backbone='CNN') -> None:
        super(NNCLR, self).__init__()
        assert backbone in ['CNN', 'Transformer', 'DeepConvLSTM'], 'Backbone type not supported now'

        self.args = ArgHolder(
            n_epoch=CL_EPOCHS,
            batch_size=32,
            framework="nnclr",
            model_name='FCN' if backbone=='CNN' else backbone,
            criterion="NTXent",
            n_class = np.nanmax(y)+1,
        )
        if self.args.model_name == "DeepConvLSTM" : self.args.model_name = 'DCL'
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
            logger=LOG,
            fitlog=self.fitlog,
            DEVICE=device,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            criterion=self.criterion,
            args=self.args
        )
        self.model.load_state_dict(best_model)
        self.model = trainer.lock_backbone(self.model, self.args)
        return

    def __del__(self):
        trainer.delete_files(self.args)
    

class NNCLR_C(NNCLR):
    def __init__(self, X, y=None) -> None:
        super(NNCLR_C, self).__init__(X, y, 'CNN')

    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model(x)
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
                _, f = self.model(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return fet.detach().cpu().numpy()
        else:
            return fet.detach().numpy()

class NNCLR_R(NNCLR):
    def __init__(self, X, y=None) -> None:
        super(NNCLR_R, self).__init__(X, y, 'CNN')

    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                _, f = self.model(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return np.nanmax(fet.detach().cpu().numpy(), axis=2)
        else:
            return np.nanmax(fet.detach().numpy(), axis=2)

class Supervised_C(nn.Module):
    def __init__(self, X, y=None) -> None:
        super(Supervised_C, self).__init__()
        self.encoder = nn.Sequential(nn.Conv1d(X.shape[1], 32, kernel_size=8, stride=1, bias=False, padding=4),
                                        nn.BatchNorm1d(32),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                        nn.Dropout(0.35),
                                        nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
                                        nn.BatchNorm1d(64),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                        nn.Conv1d(64, EMBEDDING_WIDTH, kernel_size=8, stride=1, bias=False, padding=4),
                                        nn.BatchNorm1d(EMBEDDING_WIDTH),
                                        nn.ReLU(),
                                        nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                                        nn.Flatten(),
                                        nn.AdaptiveAvgPool1d(EMBEDDING_WIDTH)                        
            )
        
        self.n_classes = np.nanmax(y) + 1
        self.classifier_block = nn.Linear(EMBEDDING_WIDTH, out_features=self.n_classes)

        self.model = nn.Sequential(self.encoder, self.classifier_block)
        self.model = self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        self.args = ArgHolder(100, 32, "", "", "", self.n_classes)

    def fit(self, X_train, y_train=None, X_val=None, y_val=None) -> None:
        """
        Train cycle with early stopping
        """
        train_loader = setup_dataloader(X_train, y_train, self.args)
        val_loader = setup_dataloader(X_val, y_val, self.args)
        es = EarlyStopping(tolerance=10, min_delta=0.001)
        for epoch in range(self.args.n_epoch):
            print(f'Epoch {epoch}:')
            total_loss = 0
            val_loss = 0
            for i, (x0, y0, d) in enumerate(train_loader):
                self.optimizer.zero_grad()
                x0 = x0.to(device)
                y0 = y0.type(torch.LongTensor)
                y0 = y0.to(device)
                if x0.size(0) != self.args.batch_size:
                    continue
                f0 = self.encoder(x0)
                out = self.classifier_block(f0)
                loss = self.criterion(out, y0)
                total_loss += loss.item()
                total_loss /= self.args.batch_size
                loss.backward()
                self.optimizer.step()
                if i%4 == 0: print('.', end='')
            print('\n')
            with torch.no_grad():
                for (x1, y1, d) in val_loader:
                    x1 = x1.to(device)
                    y1 = y1.type(torch.LongTensor)
                    y1 = y1.to(device)
                    f1 = self.encoder(x1)
                    y_pred = self.classifier_block(f1)
                    loss = self.criterion(y_pred, y1)
                    val_loss += loss.item()
                    val_loss /= self.args.batch_size
            es(val_loss)
            if es.early_stop:
                print(f'Stopping early at epoch {epoch}')
                return

            print('Train loss: ', total_loss)
            print('Validation loss: ', val_loss)




    def get_features(self, X) -> np.ndarray:
        dataloader = setup_dataloader(X, np.zeros(X.shape[0]), self.args)
        fet = None
        with torch.no_grad():
            for x, y, d in dataloader:
                x = x.to(device).float()
                f = self.encoder(x)
                if fet is None:
                    fet = f
                else:
                    fet = torch.cat((fet, f))
        if device == 'cuda':
            return fet.detach().cpu().numpy()
        else:
            return fet.detach().numpy()

