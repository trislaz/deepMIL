from torch.nn import (Linear, Module, Sequential, LeakyReLU, Tanh, Softmax,
                      Sigmoid)
from torch.optim import Adam
from tensorboardX import SummaryWriter
import torch

## Use Cross_entropy loss nn.CrossEntropyLoss
# TODO change the get_* functions with _get_*

class AttentionMILFeatures(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    """
    def __init__(self, args):
        super(AttentionMILFeatures, self).__init__()
        self.feature_extractor = Sequential(
            Linear(args.feature_depth,128), 
            LeakyReLU(0.2),
            Linear(128, 128),
            LeakyReLU(0.2)
        )
        self.weight_extractor = Sequential(
            Linear(128, 64),
            Tanh(),
            Linear(64, 1),
            Softmax(dim=-2) # Softmax sur toutes les tuiles. somme à 1.
        )
        self.classifier = Sequential(
            Linear(128, 64),
            Linear(64, 1),
            Sigmoid()
        )
    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patches
        """
        f = self.feature_extractor(x)
        w = self.weight_extractor(f)
        w = torch.transpose(w, -1, -2)
        slide = torch.matmul(w, f) # Slide representation, weighted sum of the patches
        out = self.classifier(slide)
        return out

