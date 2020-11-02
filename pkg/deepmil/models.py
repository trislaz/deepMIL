"""
implementing models. DeepMIL implements a models that classify a whole slide image
"""

from torch.nn import BCELoss, NLLLoss, MSELoss
from torch.optim import Adam
import torch
import numpy as np
from sklearn import metrics
from .networks import AttentionMILFeatures, model1S, Conan, MILGene
from .dataloader import Dataset_handler
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
    def __init__(self, error_table, device, C, target_correspondance):
        """
        error_table gives the path to the error table, or None
        C  is int, gives the number of classes
        """
        error_table = self._extract_error_table(error_table, C, target_correspondance)
        self.error_table = torch.Tensor(error_table).to(device)
        self.target_correspondance = target_correspondance
        self.device = device
        self.C = C

    def _get_error_vectors(self, y):
        ev = [self.error_table[x,:].unsqueeze(0) for x in y]
        ev = torch.cat(ev)
        return ev.to(self.device)

    def __call__(self, x, y):
        x = torch.exp(x) # to proba
        nc = x.shape[-1] 
        true_gt = torch.LongTensor([self.target_correspondance[x] for x in y])
        y_oh = torch.nn.functional.one_hot(true_gt, self.C).to(self.device)
        e_t = self._get_error_vectors(y)
        x = x-y_oh
        x = x * e_t
        x = x ** 2
        return torch.mean(torch.mean(x, -1))

    def _extract_error_table(self, error_table, C, target_correspondance):
        if error_table is None:
            E_T = torch.ones((C,C))
        else:
            e_t = np.load(error_table) * 10
            s_et = e_t.sum(axis=0)
            E_T = e_t + (np.identity(e_t.shape[0]) * s_et / 3)
            E_T = E_T[:,target_correspondance]
        return E_T

class RegressionCostLoss:
    def __init__(self, error_table, device, C, target_correspondance):
        """
        error_table gives the path to the error table, or None
        C  is int, gives the number of classes
        """
        error_table = self._extract_error_table(error_table, C, target_correspondance)
        self.error_table = torch.Tensor(error_table).to(device)
        self.target_correspondance = target_correspondance
        self.loss = MSELoss()
        self.device = device
        self.C = C

    def _get_error_vectors(self, y):
        ev = [self.error_table[x,:].unsqueeze(0) for x in y]
        ev = torch.cat(ev)
        return ev.to(self.device)

    def _extract_error_table(self, error_table, C, target_correspondance):
        if error_table is None:
            E_T = torch.ones((C,C)) - np.identity(C)
        else:
            E_T = np.load(error_table)
        return E_T

    def __call__(self, x, y):
#        y = torch.LongTensor([self.target_correspondance[x] for x in y])
        y = self._get_error_vectors(y).to(self.device)
        loss = self.loss(x, y)
        return loss

class SmoothOneSidedLoss:
    """
    According to Chung et al. 2016.
    """
    def __init__(self, error_table, device, C, target_correspondance):
        self.device = device
        self.C = C
        self.tc = list(target_correspondance)
        self.tcr = [self.tc.index(x) for x in range(len(self.tc))]
        self.error_table = np.load(error_table) * 10
        
    def __call__(self, x, y):
        x = x[:,self.tcr] # set the 0 element as the real 0.
        true_gt = torch.LongTensor([self.tc[x] for x in y])
        z_nk = torch.nn.functional.one_hot(true_gt, self.C).to(self.device)
        z_nk[z_nk == 0] = -1
        c_nk = [torch.tensor(self.error_table[x, :]).unsqueeze(0) for x in true_gt]
        c_nk = torch.cat(c_nk, 0).to(self.device)
        delta_nk = (x - c_nk) * z_nk
        delta_nk = torch.log(1 + torch.exp(delta_nk))
        loss = delta_nk.sum()
        return loss

class CostMetric:
    def __init__(self, cost_table, target_correspondance=None):
        """
        cost_table, str, path to the cost(error) table
        target_corrspondance, list, gives correspondance between a label and its encoding.
        i.e the real label corresponding to the label 2 during training will be target_correspondance[2]
        """
        self.cost_table = np.load(cost_table)
        self.target_correspondance = target_correspondance
        if self.target_correspondance is None:
            self.target_correspondance = [0,1,2,3]

    def single_cost(self, x, y):
        x, y = self.target_correspondance[x], self.target_correspondance[y]
        cost = self.cost_table[x, y]
        return cost

    def __call__(self, X, Y):
        """
        X has to be : a vector of prediction, therefore a vector of integer.
        y has to be :same
        returns float.
        """
        XY = zip(X, Y)
        costs = []
        for x, y in XY:
            costs.append(self.single_cost(x, y))
        return 1 - np.mean(costs)
       
class DeepMIL(Model):
    """
    Class implementing a Deep MIL framwork. different Models a define upstairs
    """

    def __init__(self, args, target_correspondance=None, with_data=False, weights=None):
        """
        args containes all the info for initializing the deepmil and its dataloader.
        with_data, bool, when True : set the data_loaders according to args. When False, no data is loaded.
        order of _constructors fucntions is important.
        """
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
        optimizer = self._get_optimizer(args)
        self.optimizers = [optimizer]
        self._get_schedulers(args)
        self.train_loader, self.val_loader, self.target_correspondance = self._get_data_loaders(args, with_data)
        if target_correspondance is not None:
            self.target_correspondance = target_correspondance # C'est pour le cas test/predict, ou on utilise le target correspondance enregistré auparavant.
        self.cost_metric = self._get_cost_metric(args)
        self.criterion = self._get_criterion(args.criterion)
        self.bayes = False

    def _get_network(self):
        net = MILGene(self.args)       
        net = net.to(self.args.device)
        return net
    
    def _get_data_loaders(self, args, with_data):
        train_loader, val_loader, target_correspondance = None, None, None
        if with_data:
            data = Dataset_handler(args)
            train_loader, val_loader = data.get_loader(training=True)
            target_correspondance = train_loader.dataset.target_correspondance
        return train_loader, val_loader, target_correspondance

    def _get_cost_metric(self, args):
        cost_metric = None
        if args.error_table is not None:
            cost_metric = CostMetric(self.error_table, self.target_correspondance)
        return cost_metric
            
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
        if criterion == 'nll': # to use with attentionmilmultihead.
            criterion = NLLLoss(weight=self.weights).to(self.args.device)
        elif criterion == 'mse':
            criterion = CostSensitiveLoss(self.error_table, self.args.device, self.args.num_class, self.target_correspondance)
        elif criterion == 'regcost':
            criterion = RegressionCostLoss(self.error_table, self.args.device, self.args.num_class, self.target_correspondance)
        elif criterion == 'sosr':
            criterion = SmoothOneSidedLoss(self.error_table, self.args.device, self.args.num_class, self.target_correspondance)
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
        if self.model_name in ['multiheadmulticlass', 'mhmc_conan', 'mhmc_layers']:
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

    def _predict_function(self, scores):
        """
        depends on the framework.
        """
        if self.args.criterion == "regcost" or self.args.criterion == 'sosr':
            preds = np.argmin(scores, axis=-1)
        elif self.bayes:
            print('bayes Preds')
            if len(scores.shape) <2:
                np.expand_dims(scores, 0)
            error_table = np.load(self.error_table)
            target_correspondance = list(self.target_correspondance)
            target_correspondance_reverse = [target_correspondance.index(x) for x in range(len(target_correspondance))]
            scores = scores[:,target_correspondance_reverse]
            Pk = np.zeros(scores.shape)
            for b in range(scores.shape[0]):
                for pred in range(scores.shape[1]):
                    for i in range(scores.shape[1]):
                        Pk[b, pred] += scores[b, i] * error_table[i, pred]
            Pk = Pk[:,target_correspondance]
            preds = np.argmin(Pk, axis=-1)
        else:
            preds = np.argmax(scores, axis=-1)
        return preds

    def _compute_metrics(self, scores, y_true):
        report = metrics.classification_report(y_true=y_true, y_pred=self._predict_function(scores), output_dict=True, zero_division=0)
        metrics_dict = {'accuracy': report['accuracy'], "precision": report['weighted avg']['precision'], 
            "recall": report['weighted avg']['recall'], "f1-score": report['weighted avg']['f1-score']}
        if self.args.num_class <= 2:
            metrics_dict['roc_auc'] = metrics.roc_auc_score(y_true=y_true, y_score=scores)
        if self.cost_metric:
            metrics_dict['cost_metric'] = self.cost_metric(X=self._predict_function(scores), Y=y_true)
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
                'target_correspondance': self.train_loader.dataset.target_correspondance}
        return dictio

class DeepMILim(DeepMIL):
    def __init__(self, args):
        super(DeepMILim, self).__init__(args)
        self.feature_extractor = {}




####################
## TO ADD IN ARGS ##
####################
# args.device 
