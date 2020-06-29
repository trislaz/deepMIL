"""
Implements the networks that can be used in train. 
use of pytorch.
"""
import functools
from torch.nn import (Linear, Module, Sequential, LeakyReLU, Tanh, Softmax, Identity, MaxPool2d, Conv3d,
                      Sigmoid, Conv1d, Conv2d, ReLU, Dropout, BatchNorm1d, BatchNorm2d, InstanceNorm1d, 
                      MaxPool3d)
import torch
from torchvision import transforms

## Use Cross_entropy loss nn.CrossEntropyLoss
# TODO change the get_* functions with _get_*
# TODO as soon as required, put the decription of args.

def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args:
        para = args.para
    else:
        para = default
    return para

class AttentionMILFeatures(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    """
    def __init__(self, args):
        super(AttentionMILFeatures, self).__init__()
        width_fe = is_in_args(args, 'width_fe', 64)
        self.feature_extractor = Sequential(
            Linear(args.feature_depth, width_fe),
            ReLU(),
            Dropout(p=args.dropout),
            Linear(width_fe, width_fe),
            ReLU(),
            Dropout(p=args.dropout)
        )
        self.weight_extractor = Sequential(
            Linear(width_fe, int(width_fe/2)),
            Tanh(),
            Linear(int(width_fe/2), 1),
            Softmax(dim=-2) # Softmax sur toutes les tuiles. somme à 1.
        )
        self.classifier = Sequential(
            Linear(width_fe, 1),
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

def get_norm_layer(use_bn=True, d=1):
    bn_dict = {1:BatchNorm1d, 2:BatchNorm2d}
    if use_bn: #Use batch
        norm_layer = functools.partial(bn_dict[d], affine=True, track_running_stats=True)
    else:
        #norm_layer = functools.partial(InstanceNorm1d, affine=False, track_running_stats=False)
        norm_layer = functools.partial(Identity)
    return norm_layer

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
        use_bn = args.constant_size & (args.batch_size > 8)
        norm_layer = get_norm_layer(use_bn)
        n_clusters = is_in_args(args, 'n_clusters', 128)
        hidden_fcn = is_in_args(args, 'hidden_fcn', 64)
        self.continuous_clusters = Sequential(
            Conv1d(in_channels=args.feature_depth, 
                   out_channels=n_clusters,
                   kernel_size=1),
            norm_layer(n_clusters),
            ReLU(),
            Dropout(p=args.dropout)
        )
        self.classifier = Sequential(
            Linear(in_features=n_clusters,
                   out_features=hidden_fcn), # Hidden_fc
            norm_layer(hidden_fcn),
            ReLU(),
            Dropout(p=args.dropout),
            Linear(in_features=hidden_fcn,
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

class Conv2d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        super(Conv2d_bn, self).__init__()
        self.norm_layer = get_norm_layer(use_bn, d=2)
        self.layer = Sequential(
            Conv3d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=(1, 3, 3),
                   padding=(0, 1, 1)),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Conv1d_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Conv1d_bn, self).__init__()
        self.layer = Sequential(
            Conv1d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=1),
            self.norm_layer(out_channels),
            ReLU(),
            Dropout(p=dropout)
        )
    def forward(self, x):
        out = self.layer(x)
        return out

class Dense_bn(Module):
    def __init__(self, in_channels, out_channels, dropout, use_bn):
        self.norm_layer = get_norm_layer(use_bn)
        super(Dense_bn, self).__init__()
        self.layer = Sequential(
            Linear(in_features=in_channels, 
                   out_features=out_channels),
            self.norm_layer(out_channels),
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
        self.hidden1 = is_in_args(args, 'hidden1', 32)
        self.hidden2 = int(hidden1/4) 
        self.hidden_fcn = is_in_args(args, 'hidden_fcn', 32)
        use_bn = args.constant_size & (args.batch_size > 8)
        super(Conan, self).__init__()
        self.continuous_clusters = Sequential(
            Conv1d_bn(in_channels=args.feature_depth,
                      out_channels=hidden1, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=hidden1,
                      out_channels=hidden2, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=hidden2,
                      out_channels=hidden1, 
                      dropout=args.dropout,
                      use_bn=use_bn)
        )
        self.weights = Sequential(
            Conv1d(in_channels=hidden1, 
                   out_channels=1,
                   kernel_size=1),
            ReLU()
        )
        self.classifier = Sequential(
            Dense_bn(in_channels=(hidden1 + 1) * 2 * self.k + hidden1,
                     out_channels=hidden_fcn,
                     dropout=args.dropout, 
                     use_bn=use_bn),
            Dense_bn(in_channels=hidden_fcn,
                     out_channels=hidden_fcn, 
                     dropout=args.dropout,
                     use_bn=use_bn),
            Linear(in_features=hidden_fcn,
                   out_features=1),
            Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.continuous_clusters(x)
        scores = self.weights(out)
        _, indices = torch.sort(scores, dim=-1)

        ## Aggregation
        selection = torch.cat((indices[:, :, :self.k], indices[:, :, -self.k:]), axis=-1)
        selection_out = torch.cat([selection] * self.hidden1, axis=1)
        out = torch.gather(out, -1, selection_out)
        scores = torch.gather(scores, -1, selection)
        avg = torch.mean(out, dim=-1)

        out = torch.cat((scores.flatten(1, -1), avg.flatten(1, -1), out.flatten(1, -1)), axis=-1)
        out = self.classifier(out)
        out = out.squeeze(-1)
        return out
    
class FeatureExtractor(Module):
    def __init__(self, in_shape, out_shape, dropout, use_bn):
        super(FeatureExtractor, self).__init__()
        self.in_dense = int(32 * ((in_shape/4)**2))
        self.conv_layers = Sequential(
           Conv2d_bn(3, 32, dropout, use_bn),
           Conv2d_bn(32, 64, dropout, use_bn),
           MaxPool3d((1, 2, 2)),
           Conv2d_bn(64, 32, dropout, use_bn),
           MaxPool3d((1, 2, 2)),
           Conv2d_bn(32, 32, dropout, use_bn))
        self.dense_layer = Dense_bn(self.in_dense, out_shape, dropout, use_bn)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.flatten(2, -1)
        x = self.dense_layer(x)
        return x

class MILfromIm(Module):
    models = {'attentionmil': AttentionMILFeatures, 
                'conan': Conan, 
                '1s': model1S 
                }
    def __init__(self, args):
        super(MILfromIm, self).__init__()
        self.feature_extractor = FeatureExtractor(in_shape=args.in_shape,
                                                  out_shape=args.feature_depth,
                                                  dropout=args.dropout,
                                                  use_bn=False)
        self.mil = self.models[args.model_name](args)
    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.mil(x)
        return x

if __name__ == '__main__':
    import numpy as np
    from argparse import Namespace
    #slide = np.load('/Users/trislaz/Documents/cbio/data/tcga/tcga_all_encoded/mat_pca/image_tcga_0.npy')
    slide = torch.ones((8, 3, 16, 32, 32))
    slide = torch.FloatTensor(slide)
    args = {'feature_depth': 256,
            'dropout':0,
            'in_shape': 32,
            'model_name': 'conan',
            'constant_size':True,
            'batch_size': 16
            }
    args = Namespace(**args)
    model = MILfromIm(args)
    model.eval()
    output = model(slide)
    classif_score = output