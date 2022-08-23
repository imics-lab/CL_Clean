#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 23 Aug, 2022
#
#Make some nice models with a common interface

from ..CL_HAR.models import backbones, attention, frameworks
import torch
from torch import nn

EMBEDDING_WIDTH = 64

class Conv_Autoencoder(nn.Module):
    def __init__(self, X) -> None:
        pass