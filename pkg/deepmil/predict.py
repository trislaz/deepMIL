"""Just predicts, given a set of WSI.
"""
import numpy as np
import pandas as pd
import torch
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
        x = x.unsqueeze(0)
        y_hat = model.predict(x)
        id_im = os.path.splitext(os.path.basename(dataloader.dataset.files[o]))[0] 
        subtype = 0#df[df['ID'] == id_im]['Subtype_NEW'].item()
        test = df[df['ID'] == id_im]['test'].item()
        success = y_hat[1].item() == y.item()
        results.append({'prediction': y_hat[1].item(), 'gt': y.item(), 'index':o, 'subtype':subtype, 'ID':id_im, 'test':test, 'success': success})
    return results, dataloader
