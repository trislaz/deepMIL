"""
implementing models. DeepMIL implements a models that classify a whole slide image
"""

from torch.nn import BCELoss, NLLLoss
from torch.optim import Adam
import torch
import numpy as np
from sklearn import metrics
from .networks import AttentionMILFeatures, model1S, Conan, MILGene
from .model_base import Model
# For the sklearn warnings

## Use Cross_entropy loss nn.CrossEntropyLoss
# TODO change the get_* functions with _get_*
# TODO better organizing in the class the different metrics writing (losses, mean losses
# # classif metrics...)
# In theory, the writer should be written in the base_class as it is not dependant on the model used
# we should be able to pass him the dict object containing what has to be written.
# maybe use a list containing all the dict at all epochs: writes only the last object.
# 
# TODO As in dataloader.py, make a list of the arguments that have to be in the Namespace.
##

class CostSensitiveLoss:
    def __init__(self, error_table, device, C):
        """
        error_table gives the path to the error table, or None
        C  is int, gives the number of classes
        """
        error_table = self._extract_error_table(error_table, C)
        self.error_table = torch.Tensor(error_table).to(device)
        self.device = device
        self.C = C

    def _get_error_vectors(self, y):
        ev = [self.error_table[x,:].unsqueeze(0) for x in y]
        ev = torch.cat(ev)
        return ev.to(self.device)

    def __call__(self, x, y):
        x = torch.exp(x) # to proba
        nc = x.shape[-1] 
        y_oh = torch.nn.functional.one_hot(y, self.C).to(self.device)
        e_t = self._get_error_vectors(y)
        x = x-y_oh
        x = x * e_t
        x = x ** 2
        return torch.mean(torch.mean(x, -1))

    def _extract_error_table(self, error_table, C):
        if error_table is None:
            E_T = torch.ones((C,C))
        else:
            e_t = np.load(error_table)
            s_et = e_t.sum(axis=0)
            E_T = e_t + np.identity(e_t.shape[0]) * s_et
        return E_T


class DeepMIL(Model):
    """
    Class implementing a Deep MIL framwork. different Models a define upstairs
    """

    def __init__(self, args, weights=None):
        super(DeepMIL, self).__init__(args)
        self.results_val = {'scores': [],
                            'y_true': []}
        self.mean_train_loss = 0
        self.mean_val_loss = 0
        self.target_correspondance = [] # Useful when writing the results
        self.weights = weights
        self.model_name = args.model_name
        self.network = self._get_network()
        self.error_table = args.error_table
        self.criterion = self._get_criterion(args.criterion)
        optimizer = self._get_optimizer(args)
        self.optimizers = [optimizer]
        self._get_schedulers(args)

    def _get_network(self):
        net = MILGene(self.args)       
        net = net.to(self.args.device)
        return net


    def _get_schedulers(self, args):
        """Can be called after having define the optimizers (list-like)
        """
        if args.lr_scheduler == 'linear':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
            self.schedulers = [scheduler(optimizer=o, patience=self.args.patience_lr, factor=0.3) for o in self.optimizers]
        if args.lr_scheduler == 'cos':
            self.schedulers = [torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=o, T_0=1, T_mult=2) for o in self.optimizers]

    def _get_optimizer(self, args):
        if args.optimizer == 'adam':
            optimizer = Adam(self.network.parameters(), lr=args.lr)
        if args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.network.parameters(), args.lr,
                                momentum=0.9,
                                weight_decay=1e-4)
        return optimizer

    def _get_criterion(self, criterion):
        if criterion == 'bce':
            criterion = BCELoss(weight=self.weights).to(self.args.device)
        if criterion == 'nll':
            criterion = NLLLoss(weight=self.weights).to(self.args.device)
        elif criterion == 'mse':
            criterion = CostSensitiveLoss(self.error_table, self.args.device, self.args.num_class)
        return criterion
    
    def _forward_no_grad(self, x):
        with torch.no_grad():
            out = self.network(x)
        out = out.detach()
        return out

    def _to_pseudo_proba(self, out):
        """
        Transform the output of the network into a pseudo-proba.
        Depends on the network used : multiheadmulticlass is ended with a 
        LOGsoftmax, the others directly with the softmax.
        """
        if self.model_name in ['multiheadmulticlass', 'mhmc_conan']:
            return np.exp(out)
        else:
            return out

    def _keep_best_metrics(self, metrics):
        factor = self.args.sgn_metric 
        if self.best_ref_metric is None:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        if self.best_ref_metric * factor > metrics[self.ref_metric] * factor:
            print('old acc : {}, new acc : {}'.format(self.best_ref_metric, metrics[self.ref_metric]))
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        
    def flush_val_metrics(self):
        val_scores = np.array(self.results_val['scores'])
        val_y = np.array(self.results_val['y_true'])
        val_metrics = self._compute_metrics(scores=val_scores, y_true=val_y)
        val_metrics['mean_train_loss'] = self.mean_train_loss
        val_metrics['mean_val_loss'] = self.mean_val_loss
        self._keep_best_metrics(val_metrics)

        # Re Initialize val_results for next validation
        self.results_val['scores'] = []
        self.results_val['y_true'] = []
        return val_metrics

    def _compute_metrics(self, scores, y_true):
        report = metrics.classification_report(y_true=y_true, y_pred=np.argmax(scores, axis=-1), output_dict=True, zero_division=0)
        metrics_dict = {'accuracy': report['accuracy'], "precision": report['weighted avg']['precision'], 
            "recall": report['weighted avg']['recall'], "f1-score": report['weighted avg']['f1-score']}
        if self.args.num_class <= 2:
            metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=y_true, y_score=scores)
        return metrics_dict

    def predict(self, x):
        x = x.to(self.args.device)
        proba = self._to_pseudo_proba(self._forward_no_grad(x).cpu())
        _, pred = torch.max(proba, -1)
        pred = self.target_correspondance[int(pred.item())]
        return proba.numpy(), pred

    def evaluate(self, x, y):
        """
        takes x, y torch.Tensors.
        Predicts on x, stores y and the loss.
        """
        y = y.to(self.args.device, dtype=torch.int64)
        x = x.to(self.args.device)
        scores = self._forward_no_grad(x)
        loss = self.criterion(scores, y)       
        y = y.to('cpu', dtype=torch.int64)
        scores = scores.to('cpu')
        self.results_val['scores'] += list(self._to_pseudo_proba(scores.numpy()))
        self.results_val['y_true'] += list(y.cpu().numpy())
        return loss.detach().cpu().item()

    def forward(self, x):
        out = self.network(x)
        return out

    def optimize_parameters(self, input_batch, target_batch):
        self.set_zero_grad()
        if self.args.constant_size: # We can process a batch as a whole big tensor
            input_batch = input_batch.to(self.args.device)
            target_batch = target_batch.to(self.args.device, dtype=torch.int64)
            output = self.forward(input_batch)
            loss = self.criterion(output, target_batch)
            loss.backward()

        else: # We have to process a batch as a list of tensors (of different sizes)
            loss = 0
            for o, im in enumerate(input_batch):
                im = im.to(self.args.device)
                target = target_batch[o].to(self.args.device, dtype=torch.int64)
                output = self.forward(im)
                loss += self.criterion(output, target.float())
            loss = loss/len(input_batch)
            loss.backward()

        self.optimizers[0].step()
        return loss.detach().cpu().item()

    def make_state(self):
        dictio = {'state_dict': self.network.state_dict(),
                'state_dict_optimizer': self.optimizers[0].state_dict, 
                'state_scheduler': self.schedulers[0].state_dict(), 
                'inner_counter': self.counter,
                'args': self.args,
                'best_metrics': self.best_metrics, 
                'dataset':self.dataset}
        return dictio

class DeepMILim(DeepMIL):
    def __init__(self, args):
        super(DeepMILim, self).__init__(args)
        self.feature_extractor = {}




####################
## TO ADD IN ARGS ##
####################
# args.device 
