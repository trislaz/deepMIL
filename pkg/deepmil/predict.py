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

def load_model(config, model_path=None, dataset=None, table_data=None):
    args = get_arguments(train=False, config=config)
    if model_path is not None:
        args.model_path = model_path
    if dataset is not None:
        args.path_tiles = dataset
    if table_data is not None:
        args.table_data = table_data
    model = DeepMIL(args=args)
    state = torch.load(args.model_path, map_location='cpu')
    args.test_fold = state['args'].test_fold
    model.network.load_state_dict(state['state_dict'])
    data = Dataset_handler(args, predict=True)
    dataloader = data.get_loader(training=False)
    df = dataloader.dataset.table_data
    results = []
    model.network.eval()
    for o, (x, y) in enumerate(dataloader):
        y_hat = model.predict(x)
        id_im = os.path.splitext(os.path.basename(dataloader.dataset.files[o]))[0] 
        serie = df[df['ID'] == id_im].to_dict('records')[0]
        success = y_hat[1].item() == y.item()
        r = {'prediction': y_hat[1].item(), 'gt': y.item(), 'index':o,'success': success}
        r.update(serie) 
        results.append(r)
    results = pd.DataFrame(results)
    results_test = results[results['test'] == args.test_fold]
    predicted_labels = results_test['prediction'].values
    true_labels = results_test['gt'].values
    confusion_mat = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    return results, confusion_mat, dataloader.dataset.target_correspondance 
