from torch.nn import (Linear, Module, Sequential, LeakyReLU, Tanh, Softmax,
                      Sigmoid, Conv1d, ReLU, Dropout, BatchNorm1d)
from torch.optim import Adam
from tensorboardX import SummaryWriter
import torch
from torchvision import transforms

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
            Dropout(p=args.dropout),
            Linear(128, 128),
            LeakyReLU(0.2),
            Dropout(p=args.dropout)
        )
        self.weight_extractor = Sequential(
            Linear(128, 64),
            Tanh(),
            Linear(64, 1),
            Softmax(dim=-2) # Softmax sur toutes les tuiles. somme à 1.
        )
        self.classifier = Sequential(
            Linear(128, 64),
            ReLU(),
            Dropout(p=args.dropout),
            Linear(64, 1),
            Dropout(p=args.dropout),
            Sigmoid()
        )
    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        f = self.feature_extractor(x)
        w = self.weight_extractor(f)
        w = torch.transpose(w, -1, -2)
        slide = torch.matmul(w, f) # Slide representation, weighted sum of the patches
        out = self.classifier(slide)
        out = out.squeeze(-1).squeeze(-1)
        return out


class model1S(Module):
    """
    Args must contain : 
        * feature_depth: int, number of features of the inuput
        * dropout: float, dropout parameter
    
    Ends without a Softmax, to use BCEWithLogitsLoss
    Takes as input a WSI as a Tensor of shape BxNxD where :
        * D the feature depth
        * N the number of tiles
        * B the batch dimension

    The first operation is to transform the tensor in the form (B)xDxN 
    so that D is the 'channel' dimension

    """
    def __init__(self, args):
        super(model1S, self).__init__()
        self.continuous_clusters = Sequential(
            Conv1d(in_channels=args.feature_depth, 
                   out_channels=128,
                   kernel_size=1),
            BatchNorm1d(128),
            ReLU(),
            Dropout(p=args.dropout)
        )
        self.classifier = Sequential(
            Linear(in_features=128,
                   out_features=64), # Hidden_fc
            BatchNorm1d(64),
            ReLU(),
            Dropout(p=args.dropout),
            Linear(in_features=64,
                   out_features=1),
            Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.continuous_clusters(x)
        out = out.max(dim=-1)[0] # AvgPooling
        out = self.classifier(out)
        out = out.squeeze(-1)
        return out


if __name__ == '__main__':
    import numpy as np
    from argparse import Namespace
    slide = np.load('minimal_example/numpy_WSI/311848.npy')
    slide = torch.FloatTensor(slide)
    slide = slide.unsqueeze(0)
    args = Namespace(feature_depth=2048, dropout=0.2)
    model = model1S(args)
    model.eval()
    output = model(slide)

    classif_score = output