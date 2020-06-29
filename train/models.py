"""
implementing models. DeepMIL implements a models that classify a whole slide image
"""
from torch.nn import BCELoss
from torch.optim import Adam
import torch
import numpy as np
from sklearn import metrics
from networks import AttentionMILFeatures, model1S, Conan
from model_base import Model
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

class DeepMIL(Model):
    """
    Class implementing a Deep MIL framwork. different Models a define upstairs
    """
    models = {'attentionmil': AttentionMILFeatures, 
                'conan': Conan, 
                '1s': model1S}

    def __init__(self, args):
        super(DeepMIL, self).__init__(args)
        self.results_val = {'scores': [],
                            'y_true': []}
        self.mean_train_loss = 0
        self.mean_val_loss = 0
        self.target_correspondance = [] # Useful when writing the results
        self.network = self._get_network()
        self.criterion = BCELoss()
        optimizer = Adam(self.network.parameters(), lr=args.lr)
        self.optimizers = [optimizer]
        self.get_schedulers()

    def _get_network(self):
        net = self.models[self.args.model_name](self.args)        
        net = net.to(self.args.device)
        return net
    
    def _forward_no_grad(self, x):
        with torch.no_grad():
            out = self.network(x)
        out = out.detach().cpu()
        return out

    def _keep_best_metrics(self, metrics):
        factor = {'accuracy': -1, 'loss': 1}
        factor = factor[self.ref_metric]
        if self.best_ref_metric is None:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        if self.best_ref_metric * factor > metrics[self.ref_metric] * factor:
            self.best_ref_metric = metrics[self.ref_metric]
            self.best_metrics = metrics
        
    def flush_val_metrics(self):
        val_scores = np.array(self.results_val['scores'])
        val_y = np.array(self.results_val['y_true'])
        val_metrics = self._compute_metrics(scores=val_scores, y_true=val_y)
        self._keep_best_metrics(val_metrics)
        val_metrics['mean_loss'] = {'mean_train_loss': self.mean_train_loss,
                                    'mean_val_loss': self.mean_val_loss}

        # Re Initialize val_results for next validation
        self.results_val['scores'] = []
        self.results_val['y_true'] = []
        return val_metrics

    def _compute_metrics(self, scores, y_true):
        report = metrics.classification_report(y_true=y_true, y_pred=scores.round(), output_dict=True)
        metrics_dict = {'accuracy': report['accuracy'], "precision": report['weighted avg']['precision'], 
            "recall": report['weighted avg']['recall'], "f1-score": report['weighted avg']['f1-score']}
        metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=y_true, y_score=scores)
        return metrics_dict

    def predict(self, x):
        x = x.to(self.args.device)
        proba = self._forward_no_grad(x)
        pred = proba.round()
        return proba.numpy(), pred.numpy()

    def evaluate(self, x, y):
        """
        takes x, y torch.Tensors.
        Predicts on x, stores y and the loss.
        """
        y = y.to(self.args.device)
        x = x.to(self.args.device)
        scores = self._forward_no_grad(x)
        y = y.to('cpu')
        loss = self.criterion(scores, y)       
        self.results_val['scores'] += list(scores.numpy())
        self.results_val['y_true'] += list(y.cpu().numpy())
        return loss.detach().cpu().item()

    def forward(self, x):
        out = self.network(x)
        return out

    def optimize_parameters(self, input_batch, target_batch):
        self.set_zero_grad()
        if self.args.constant_size: # We can process a batch as a whole big tensor
            input_batch = input_batch.to(self.args.device)
            target_batch = target_batch.to(self.args.device)
            output = self.forward(input_batch)
            loss = self.criterion(output, target_batch)
            loss.backward()

        else: # We have to process a batch as a list of tensors (of different sizes)
            loss = 0
            for o, im in enumerate(input_batch):
                im = im.to(self.args.device)
                target = target_batch[o].to(self.args.device)
                output = self.forward(im)
                loss += self.criterion(output, target)
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
                'best_metrics': self.best_metrics}
        return dictio

class DeepMILim(DeepMIL):
    def __init__(self, args):
        super(DeepMILim, self).__init__(args)
        self.feature_extractor = {}




####################
## TO ADD IN ARGS ##
####################
# args.device 
