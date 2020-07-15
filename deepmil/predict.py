"""Just predicts, given a set of WSI.
"""
import numpy as np
import pandas as pd
import torch
from torch import load
from arguments import get_arguments
from models import DeepMIL
from dataloader import EmbededWSI
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
    dataloader = EmbededWSI(args, predict=True)
    results = []
    model.network.eval()
    for o, (x, y) in enumerate(dataloader):
        x = x.unsqueeze(0)
        y_hat = model.predict(x)
        results.append({'prediction': y_hat[1].item(), 'gt': y.item(), 'index':o})
    return results, dataloader