# Local import
from model_base import Model
from networks import AttentionMILFeatures
from torch.nn import (BCELoss, functional)
from torch.optim import Adam
from tensorboardX import SummaryWriter
import torch
from sklearn import metrics
## Use Cross_entropy loss nn.CrossEntropyLoss
# TODO change the get_* functions with _get_*
# TODO better organizing in the class the different metrics writing (losses, mean losses, classif metrics...)
# TODO In theory, the writer should be written in the base_class as it is not dependant on the model used.
# TODO we should be able to pass him the dict object containing what has to be written.
# TODO maybe use a list containing all the dict at all epochs: writes only the last object.
# TODO 
# TODO As in dataloader.py, make a list of the arguments that have to be in the Namespace.
##

class DeepMIL(Model):
    """
    Class implementing a Deep MIL framwork. different Models a define upstairs
    """
    models = {'attentionmil': AttentionMILFeatures, 
                'conan': 'not yet implemented', 
                '1s': 'not yet implemented'}

    def __init__(self, args):
        super(DeepMIL, self).__init__(args)
        self.network = self._get_network()
        self.criterion = BCELoss()
        optimizer = Adam(self.network.parameters())
        self.optimizers = [optimizer]
        self.writer = SummaryWriter(self.get_folder_writer())
        self.get_schedulers
        self.results_val = {'scores': [],
                            'y_true': []}
        self.mean_train_loss = 0
        self.mean_val_loss = 0
        self.target_correspondance = [] # Useful when writing the results

    def _get_network(self):
        net = self.models[self.args.model_name](self.args)        
        net = net.to(self.args.device)
        return net
    
    def _forward_no_grad(self, x):
        with torch.no_grad():
            out = self.network(x)
        out = out.detach().cpu()
        return out

    def flush_val_metrics(self):
        val_scores = self.results_val['scores']
        val_y = self.results_val['y_true']
        val_metrics = self._compute_metrics(scores=val_scores, y_true=val_y)
        val_metrics['mean_loss'] = {'mean_train_loss': self.mean_train_loss,
                                    'mean_val_loss': self.mean_val_loss}

        # Ugly. Changes the name of the key according to the target correspondance.
        val_metrics[self.target_correspondance[0]] = val_metrics['0']
        val_metrics[self.target_correspondance[1]] = val_metrics['1']
        del val_metrics['0']
        del val_metrics['1']
        # Re Initialize val_results for next validation
        self.results_val['scores'] = []
        self.results_val['y_true'] = []
        return val_metrics

    def _compute_metrics(self, scores, y_true):
        metrics_dict = metrics.classification_report(y_true=y_true, y_pred=scores.round())
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
        self.results_val['scores'] += scores.numpy()
        self.results_val['y_true'] += y.cpu().numpy()
        return loss.detach().cpu().item()

    def forward(self, x):
        out = self.network(x)
        return out

    def optimize_parameters(self, input_batch, target_batch):
        input_batch = input_batch.to(self.args.device)
        target_batch = target_batch.to(self.args.device)
        output_batch = self.forward(input_batch).squeeze()
        self.set_zero_grad()
        loss = self.criterion(output_batch, target_batch)
        loss.backward()
        return loss.detach().cpu().item()

    def make_state(self):
        dictio = {'state_dict': self.network.state_dict(),
                'state_dict_optimizer': self.optimizers[0].state_dict, 
                'state_scheduler': self.schedulers[0].state_dict(), 
                'inner_counter': self.counter,
                'args': self.args}
        return dictio






####################
## TO ADD IN ARGS ##
####################
# args.device 
