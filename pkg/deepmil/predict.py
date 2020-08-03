"""Just predicts, given a set of WSI.
"""
import numpy as np
import pandas as pd
import torch
import pandas as pd
from torch import load
import os
from .arguments import get_arguments
from .models import DeepMIL
from .dataloader import EmbededWSI, Dataset_handler
from collections import MutableMapping

def update_confusion(confu_dict, y_hat, y):
    if y == 1:
        if y_hat == 1:
            confu_dict['TP'] += 1
        else:
            confu_dict['FN'] += 1
    else:
        if y_hat == 1:
            confu_dict['FP'] += 1
        else:
            confu_dict['TN'] += 1
    return confu_dict


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
    confusion_dict = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    results = []
    model.network.eval()
    for o, (x, y) in enumerate(dataloader):
        y_hat = model.predict(x)
        id_im = os.path.splitext(os.path.basename(dataloader.dataset.files[o]))[0] 
        serie = df[df['ID'] == id_im].to_dict('records')[0]
        success = y_hat[1].item() == y.item()
        results.append({'prediction': y_hat[1].item(), 'gt': y.item(), 'index':o,'success': success}.update(serie))
        if args.test_fold == serie['test']:
            confusion_dict = update_confusion(confu_dict = confusion_dict, y_hat=y_hat, y=y)
    return pd.DataFrame(results), confusion_dict
