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
            Linear(args.feature_depth,64), 
            ReLU(0.2),
            Dropout(p=args.dropout),
            Linear(64, 64),
            ReLU(0.2),
            Dropout(p=args.dropout)
        )
        self.weight_extractor = Sequential(
            Linear(64, 32),
            Tanh(),
            Linear(32, 1),
            Softmax(dim=-2) # Softmax sur toutes les tuiles. somme à 1.
        )
        self.classifier = Sequential(
            Linear(64, 1),
#            ReLU(),
#            Dropout(p=args.dropout),
#            Linear(64, 1),
#            Dropout(p=args.dropout),
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
#        out = torch.nn.functional.sigmoid(out)
        return out


class model1S(Module):
    """
    Args must contain : 
        * feature_depth: int, number of features of the inuput
        * dropout: float, dropout parameter
    
    Ends witha Softmax, to use BCELoss 
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

class Conv1d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Conv1d_bn, self).__init__()
        self.layer = Sequential(
            Conv1d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=1),
            BatchNorm1d(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Dense_bn(Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(Dense_bn, self).__init__()
        self.layer = Sequential(
            Linear(in_features=in_channels, 
                   out_features=out_channels),
            BatchNorm1d(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out



class Conan(Module):
    """
    Args must contain : 
        * feature_depth: int, number of features of the inuput
        * dropout: float, dropout parameter
    
    Ends witha Softmax, to use BCELoss 
    Takes as input a WSI as a Tensor of shape BxNxD where :
        * D the feature depth
        * N the number of tiles
        * B the batch dimension

    The first operation is to transform the tensor in the form (B)xDxN 
    so that D is the 'channel' dimension

    """
    def __init__(self, args):
        self.k = 10
        self.hidden1 = 32
        self.hidden2 = 8
        self.hidden_fcn = 32
        super(Conan, self).__init__()
        self.continuous_clusters = Sequential(
            Conv1d_bn(in_channels=args.feature_depth,
                      out_channels=self.hidden1, 
                      dropout=args.dropout),
            Conv1d_bn(in_channels=self.hidden1,
                      out_channels=self.hidden2, 
                      dropout=args.dropout),
            Conv1d_bn(in_channels=self.hidden2,
                      out_channels=self.hidden1, 
                      dropout=args.dropout)
        )
        self.weights = Sequential(
            Conv1d(in_channels=self.hidden1, 
                   out_channels=1,
                   kernel_size=1),
            BatchNorm1d(1),
            ReLU()
        )
        self.classifier = Sequential(
            Dense_bn(in_channels=(self.hidden1 + 1) * 2 * self.k + self.hidden1,
                     out_channels=self.hidden_fcn,
                     dropout=args.dropout),
            Dense_bn(in_channels=self.hidden_fcn,
                     out_channels=self.hidden_fcn, 
                     dropout=args.dropout),
            Linear(in_features=self.hidden_fcn,
                   out_features=1),
            Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.continuous_clusters(x)
        scores = self.weights(out)
        _, indices = torch.sort(scores, dim=-1)

        ## Aggregation
        selection = torch.cat((indices[:, :,  :self.k], indices[:, :, -self.k:]), axis=-1)
        selection_out = torch.cat([selection] * self.hidden1, axis=1)
        out = torch.gather(out, -1, selection_out)
        scores = torch.gather(scores, -1, selection)
        avg = torch.mean(out, dim=-1)

        out = torch.cat((scores.flatten(1, -1), avg.flatten(1, -1), out.flatten(1, -1)), axis=-1)
        out = self.classifier(out)
        out = out.squeeze(-1)
        return out
    


if __name__ == '__main__':
    import numpy as np
    from argparse import Namespace
    slide = np.load('/Users/trislaz/Documents/cbio/data/tcga/tcga_all_encoded/mat_pca/image_tcga_0.npy')
    slide = torch.FloatTensor(slide)
    slide = slide.unsqueeze(0)
    args = Namespace(feature_depth=2048, dropout=0)
    model = Conan(args)
    model.eval()
    output = model(slide)
    classif_score = output