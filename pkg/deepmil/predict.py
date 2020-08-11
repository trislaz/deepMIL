"""Just predicts, given a set of WSI.
"""
import numpy as np
import seaborn as sns
from sklearn import metrics
import pandas as pd
import torch
import pandas as pd
from torch import load
import os
from .arguments import get_arguments
from .models import DeepMIL
from .dataloader import EmbededWSI, Dataset_handler
from collections import MutableMapping

def load_model(model_path):
    """Loads and prepare a learned model for prediction.

    Args:
        model_path (str): path to the *.pt.tar model
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']
    args.target_correspondance = checkpoint['dataset'].target_correspondance
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DeepMIL(args)
    model.network.load_state_dict(checkpoint['state_dict'])
    model.network.eval()
    return model

def predict(model_path=None):
    model = load_model(model_path)
    data = Dataset_handler(model.args, predict=True)
    dataloader = data.get_loader(training=False)
    df = dataloader.dataset.table_data
    results = []
    model.network.eval()
    for o, (x, y) in enumerate(dataloader):
        y_hat = model.predict(x)
        id_im = os.path.splitext(os.path.basename(dataloader.dataset.files[o]))[0].split('_embedded')[0]
        serie = df[df['ID'] == id_im].to_dict('records')[0]
        success = y_hat[1].item() == y.item()
        r = {'prediction': y_hat[1].item(), 'gt': y.item(), 'index':o,'success': success}
        r.update(serie) 
        results.append(r)
    results = pd.DataFrame(results)
    results_test = results[results['test'] == model.args.test_fold]
    predicted_labels = results_test['prediction'].values
    true_labels = results_test['gt'].values
    confusion_mat = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    return results, confusion_mat, dataloader.dataset.target_correspondance 
