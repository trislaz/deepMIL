"""
Implements the networks that can be used in train. 
use of pytorch.
"""
import functools
from torch.nn import (Linear, Module, Sequential, LeakyReLU, Tanh, Softmax, Identity, MaxPool2d, Conv3d,
                      Sigmoid, Conv1d, Conv2d, ReLU, Dropout, BatchNorm1d, BatchNorm2d, InstanceNorm1d, 
                      MaxPool3d, functional, LayerNorm, MultiheadAttention, LogSoftmax)
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer
from torch.nn.parameter import Parameter
import torch
from torch.nn.init import (xavier_normal_, xavier_uniform_, constant_)
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import torchvision

def deprecate_in_favor_of(new_name):
    def deprecated(func):
        def decorator(*args, **kwargs):
            print('Deprecated. Use {} instead.'.format(new_name))
            func(*args, **kwargs)
        return decorator
    return deprecated
## Use Cross_entropy loss nn.CrossEntropyLoss
# TODO change the get_* functions with _get_*

# TODO as soon as required, put the decription of args.

def is_in_args(args, name, default):
    """Checks if the parammeter is specified in the args Namespace
    If not, attributes him the default value
    """
    if name in args:
        para = getattr(args, name)
    else:
        para = default
    return para

class MultiHeadAttention(Module):
    """
    Implements the multihead attention mechanism used in 
    MultiHeadedAttentionMIL. 
    Input (batch, nb_tiles, features)
    Output (batch, nb_tiles, nheads)
    """
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.dropout = args.dropout
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.atn_layer_1_weights = Parameter(torch.Tensor(atn_dim, args.feature_depth))
        self.atn_layer_2_weights = Parameter(torch.Tensor(1, 1, self.num_heads, self.dim_heads, 1))
        self.atn_layer_1_bias = Parameter(torch.empty((atn_dim)))
        self.atn_layer_2_bias = Parameter(torch.empty((1, self.num_heads, 1, 1)))
        self._init_weights()

    def _init_weights(self):
        xavier_uniform_(self.atn_layer_1_weights)
        xavier_uniform_(self.atn_layer_2_weights)
        constant_(self.atn_layer_1_bias, 0)
        constant_(self.atn_layer_2_bias, 0)

    def forward(self, x):
        """ Extracts a series of attention scores.

        Args:
            x (torch.Tensor): size (batch, nb_tiles, features)

        Returns:
            torch.Tensor: size (batch, nb_tiles, nb_heads)
        """
        bs, nbt, _ = x.shape

        # Weights extraction
        x = F.linear(x, weight=self.atn_layer_1_weights, bias=self.atn_layer_1_bias)
        x = F.tanh(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.view((bs, nbt, self.num_heads, 1, self.dim_heads))
        x = torch.matmul(x , self.atn_layer_2_weights) + self.atn_layer_2_bias # 4 scores.
        x = x.view(bs, nbt, -1) # shape (bs, nbt, nheads) 
        #x = F.softmax(x, dim=-2)
        return x
 
       


class MultiHeadedAttentionMIL_multiclass(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    A utiliser avec NLLLoss
    """

    ## Change in other functions
    # atn_dim = dim attention.
    # num_heads = nombre de tete chercheuses 
    # num_class = nombre de classes. 

    def __init__(self, args):
        super(MultiHeadedAttentionMIL_multiclass, self).__init__()
        self.dropout = args.dropout
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.feature_depth = is_in_args(args, 'feature_depth', 512)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.num_class = is_in_args(args, 'num_class', 2)
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.attention = Sequential(
            MultiHeadAttention(args),
            Softmax(dim=-2)
        )

        self.classifier = Sequential(
            Linear(int(args.feature_depth * self.num_heads), width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, self.num_class),# Added 25/09
            LogSoftmax(dim=-1)
        )

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, nbt, _ = x.shape
        w = self.attention(x) # (bs, nbt, nheads)
        w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
        slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
        slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)
        out = self.classifier(slide)
        out = out.view((bs, self.num_class))
        return out

class MultiHeadMulticlassMIL_CONAN(MultiHeadedAttentionMIL_multiclass):
    """
    Instantiate a multiheadmulticlass mil-conan.
    Multihead attention mil with conan aggregation.
    Needs a 'k' inside args. giving the top-K vectors to use.
    Means that it cant be used 
    """
    def __init__(self, args):
        super(MultiHeadMulticlassMIL_CONAN, self).__init__(args)
        self.k = args.k

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, nbt, _ = x.shape
        k = np.min((nbt, self.k))
        w = self.attention(x) # (bs, nbt, nheads)
        scores, indices = torch.topk(w, k, 1) # scores, indices = (bs, k, nheads) and x = (bs, nbt, num_features)
        indices = indices.unsqueeze(-2) # adds the features dimension, for selection.
        indices = torch.cat([indices] * self.feature_depth, dim=-2) # indices = (bs, k, num_features, nheads)
        x_top = torch.cat([torch.gather(x, 1, indices[:,:,:,y]).unsqueeze(1) for y in range(self.num_heads)], 1) # slide : (bs,nheads, k, num_features) stacks the heads repr on the dim 1
        x_top = x_top.flatten(0,1) #( bs*nheads, k, num_features )

        scores = torch.transpose(scores, -1, -2)# (bs, nheads, nbt)
        scores = scores.unsqueeze(-2)
        scores = scores.flatten(0,1) # (bs*nhead, 1, k)
        scores = F.softmax(scores, dim=-1)

        slide = torch.matmul(scores, x_top)
        slide = slide.view(bs, self.num_heads, self.feature_depth)
        slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)
        out = self.classifier(slide)
        out = out.view((bs, self.num_class))
        return out

class MultiHeadedAttentionMIL(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    """

    ## Change in other functions
    # atn_dim = dim attention.
    # num_heads = nombre de tete chercheuses 

    def __init__(self, args):
        super(MultiHeadedAttentionMIL, self).__init__()
        self.dropout = args.dropout
        width_fe = is_in_args(args, 'width_fe', 64)
        atn_dim = is_in_args(args, 'atn_dim', 256)
        self.num_heads = is_in_args(args, 'num_heads', 1)
        self.feature_depth = is_in_args(args, 'feature_depth', 512)
        self.dim_heads = atn_dim // self.num_heads
        assert self.dim_heads * self.num_heads == atn_dim, "atn_dim must be divisible by num_heads"

        self.attention = Sequential(
            MultiHeadAttention(args)
#            Softmax(dim=-2)
        )

        self.classifier = Sequential(
            Linear(int(args.feature_depth * self.num_heads), width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, 1),# Added 25/09
            Sigmoid()
        )

    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """
        bs, nbt, _ = x.shape #x : (bs, nbt, nfeatures)
        w = self.attention(x) # (bs, nbt, nheads)
        w = F.softmax(x, dim=-2)
        w = torch.transpose(w, -1, -2) # (bs, nheads, nbt)
        slide = torch.matmul(w, x) # Slide representation, shape (bs, nheads, nfeatures)
        slide = slide.flatten(1, -1) # (bs, nheads*nfeatures)
        out = self.classifier(slide)
        out = out.view(bs)
        return out

class AttentionMILFeatures(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    """
    @deprecate_in_favor_of('MultiHeadedAttentionMIL')
    def __init__(self, args):
        super(AttentionMILFeatures, self).__init__()
        width_fe = is_in_args(args, 'width_fe', 64)
        #self.feature_extractor = Sequential(
        #    Linear(args.feature_depth, width_fe),
        #    ReLU(),
        #    Dropout(p=args.dropout),
        #    Linear(width_fe, width_fe),
        #    ReLU(),
        #    Dropout(p=args.dropout)
        #)
        self.weight_extractor = Sequential(
            Linear(args.feature_depth, int(width_fe/2)), #width_fe, int(width_fe/2)),
            Tanh(),
            Linear(int(width_fe/2), 1),
            Softmax(dim=-2) # Softmax sur toutes les tuiles. somme à 1.
        )
        self.classifier = Sequential(
            Linear(args.feature_depth, width_fe),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(width_fe, 1),# Added 25/09
            Sigmoid()
        )
    def forward(self, x):
        """
        Input x of size NxF where :
            * F is the dimension of feature space
            * N is number of patche
        """

        w = self.weight_extractor(x)
        w = torch.transpose(w, -1, -2)
        slide = torch.matmul(w, x) # Slide representation, weighted sum of the patches
        out = self.classifier(slide)
        out = out.squeeze(-1).squeeze(-1)

        return out

class AttentionMILFeatures_badweigths(Module):
    """
    Implements deepMIL while taking 1D vectors as instances (output of resnet for example)
    Bags are then NxF matrices, F the feature number (usually 2048), N the number of instances of the bag.
    """

    @deprecate_in_favor_of('MultiHeadedAttentionMIL')
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
            Linear(width_fe, int(width_fe/2)),
            ReLU(),# Added 25/09
            Dropout(p=args.dropout),# Added 25/09
            Linear(int(width_fe/2), 1),# Added 25/09
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
            Conv2d(in_channels=in_channels, 
                   out_channels=out_channels,
                   kernel_size=(3, 3),
                   padding=(1, 1)),
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
        self.hidden2 = self.hidden1//4 
        self.hidden_fcn = is_in_args(args, 'hidden_fcn', 32)
        use_bn = args.constant_size & (args.batch_size > 8)
        super(Conan, self).__init__()
        self.continuous_clusters = Sequential(
            Conv1d_bn(in_channels=args.feature_depth,
                      out_channels=self.hidden1, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=self.hidden1,
                      out_channels=self.hidden2, 
                      dropout=args.dropout, 
                      use_bn=use_bn),
            Conv1d_bn(in_channels=self.hidden2,
                      out_channels=self.hidden1, 
                      dropout=args.dropout,
                      use_bn=use_bn),
        )
        self.weights = Sequential(
            Conv1d(in_channels=self.hidden1, 
                   out_channels=1,
                   kernel_size=1),
            ReLU()
        )
        self.classifier = Sequential(
            Dense_bn(in_channels=(self.hidden1 + 1) * 2 * self.k + self.hidden1,
                     out_channels=self.hidden_fcn,
                     dropout=args.dropout, 
                     use_bn=use_bn),
            Dense_bn(in_channels=self.hidden_fcn,
                     out_channels=self.hidden_fcn, 
                     dropout=args.dropout,
                     use_bn=use_bn),
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
        selection = torch.cat((indices[:, :, :self.k], indices[:, :, -self.k:]), axis=-1)
        selection_out = torch.cat([selection] *self.hidden1, axis=1)
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
           MaxPool2d((2, 2)),
           Conv2d_bn(64, 32, dropout, use_bn),
           MaxPool2d((2, 2)),
           Conv2d_bn(32, 32, dropout, use_bn))
        self.dense_layer = Dense_bn(self.in_dense, out_shape, dropout, use_bn)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.permute(1, 0, 2, 3)
        x = x.flatten(1, -1)
        x = self.dense_layer(x)
        return x

class SelfAttentionMIL(Module):
    """Implemented according to Li et Eliceiri 2020
    """
    def __init__(self, args):
        super(SelfAttentionMIL, self).__init__()
        self.L = 128
        self.args = args
        self.maxmil = Sequential(
            Linear(args.feature_depth, 256),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(2*self.L), self.L),
            ReLU(),
            Dropout(args.dropout),
            Linear(self.L, 1)
            )
        self.queries = Sequential(
            Linear(args.feature_depth, int(2*self.L)),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(self.L*2), self.L)
            )
        self.visu = Sequential(
            Linear(args.feature_depth, int(self.L*2)),
            ReLU(),
            Dropout(args.dropout),
            Linear(int(self.L*2), self.L)
            )
        self.wsi_score = Sequential(
            Linear(self.L, 1)
            )
        self.classifier = Sequential(Linear(2, 1), Sigmoid())

    def forward(self, x):
        # Ingredients of the self attention
        milscores = self.maxmil(x)
        queries = self.queries(x)
        visu = self.visu(x)

        max_scores, max_indices = torch.max(milscores, dim=1)
        max_scores = max_scores.unsqueeze(-1)
        max_indices = torch.cat([max_indices] * self.L, axis=-1).unsqueeze(1) # Selects each of the 124 features that are part of the max-tile = creates a tensor of indices the shape of the queries
        max_query = torch.gather(queries, -2, max_indices)
        max_query = max_query.permute(0, 2, 1)
        sa_scores = torch.matmul(queries, max_query)
        sa_scores = sa_scores.permute(0, 2, 1)
        sa_scores = functional.softmax(sa_scores, dim=-1)
        weighted_visu = torch.matmul(sa_scores, visu)
        wsi_scores = self.wsi_score(weighted_visu)
        fused = torch.cat([max_scores, wsi_scores], axis=-2).squeeze(-1)
        x = self.classifier(fused)
        x = x.squeeze(-1)
        return x

class TransformerMIL(Module):
    def __init__(self, args):
        super(TransformerMIL, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=args.feature_depth, nhead=8, dim_feedforward=2048, dropout=args.dropout, activation="relu")
        encoder_norm = LayerNorm(args.feature_depth)
        self.attention = TransformerEncoder(encoder_layer, args.ntrans, encoder_norm)
        #self.attention1 = MultiheadAttention(args.feature_depth, 8)
        self.attention2 = MultiheadAttention(args.feature_depth, 8)
        self.classifier = Sequential(
            Linear(args.feature_depth, 1),
            Sigmoid())
        self.mil = AttentionMILFeatures(args)
    def forward(self, x):
        #x= self.attention(x)
        x, _ = self.attention2(x, x, x)
        #x = x.mean(-2)
        #x = self.classifier(x)
        #x = x.squeeze(-1)
        x = self.mil(x)
        return x

class MILGene(Module):
    models = {'attentionmil': AttentionMILFeatures, 
                'multiheadmil': MultiHeadedAttentionMIL,
                'multiheadmulticlass': MultiHeadedAttentionMIL_multiclass,
                'mhmc_conan': MultiHeadMulticlassMIL_CONAN,
                'conan': Conan, 
                '1s': model1S, 
                'sa': SelfAttentionMIL,
                'transformermil': TransformerMIL}     
    def __init__(self, args):
        feature_extractor = {1: Identity, 
                             0: self._get_features_net}
        super(MILGene, self).__init__()
        self.args = args
        self.features_instances = feature_extractor[args.embedded](args)
        self.mil = self.models[args.model_name](args)

    def forward(self, x):
        batch_size, nb_tiles = x.shape[0], x.shape[1]
        x = self._preprocess(x, self.args)
        x = self.features_instances(x)
        x = x.view(batch_size, nb_tiles, self.args.feature_depth)
        x = self.mil(x)
        return x

    def _get_features_net(self, args):
        if args.features_net == 'resnet':
            net = torchvision.models.resnet18(pretrained=True)
            net.fc = Identity()
        elif args.features_net == 'adm':
            net = FeatureExtractor(args.in_shape, args.feature_depth, args.dropout, False)
        return net

    def _preprocess(self, x, args):
        if args.embedded:
            x = x
        else:
            x = x.view(-1, 3, args.in_shape, args.in_shape)
        return x 

#for name, param in net.named_parameters():
#  # if the param is from a linear and is a bias
#  if "fc" in name and "bias" in name:
#    param.register_hook(hook_fn)

def hook(m, name):
    def hook_fn(m):
        print('______grad__ {} ________'.format(name))
        print(m)
        print('\n')
    m.register_hook(hook_fn)
    
def place_hook(net, layer_names):
    for n, p in net.named_parameters():
        for to_hook in layer_names:
            if to_hook in n:
                hook(p, n) 




if __name__ == '__main__':

    import numpy as np
    import torch
    from argparse import Namespace
    curie = '/mnt/data4/tlazard/data/curie/curie_recolo_tiled/imagenet/size_256/res_1/mat/353536B_embedded.npy'
    tcga = '/mnt/data4/tlazard/AutomaticWSI/outputs/tcga_all_auto_mask/tiling/imagenet/1/mat_pca/image_tcga_2.npy'
    #slide = torch.Tensor(np.load(tcga)).unsqueeze(0)
    batch_size = 16
    nb_tiles = 100
    feature_depth = 256
    slide = torch.rand((batch_size, nb_tiles, feature_depth))
    res = torchvision.models.resnet18()
    args = {'feature_depth': feature_depth,
            'dropout':0,
            'in_shape': 256,
            'model_name': 'multiheadmil',
            'constant_size':True,
            'features_net': 'resnet',
            'batch_size': batch_size,
            'nb_tiles' : nb_tiles,
            'embedded': 1,
            'ntrans':4,
            'num_heads':4
            }
    args = Namespace(**args)
    model = MILGene(args)
    model(slide)
    
#    to_hook =  set(['layers.0.linear1.weight', 'classifier.0.weight', 'layers.1.linear1.weight', 'layers.2.linear1.weight'])
#
#    ## Hook ! 
#    place_hook(model, to_hook)
#    slide1 = slide + 1 
#    slide1.retain_grad()
#    x = model(slide1)
#    x=x.squeeze()
#    loss = torch.nn.BCELoss()(x, torch.ones(2))
#    loss.backward()
#    print('over')
#    #model.eval()
#    #output = model(slide)
#
