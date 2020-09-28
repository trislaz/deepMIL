"""
Test a given model.
"""
import numpy as np
import pandas as pd
import torch
from torch import load
from .arguments import get_arguments
from .models import DeepMIL
from .dataloader import Dataset_handler
from .predict import load_model
from collections import MutableMapping

def convert_flatten(d, parent_key='', sep='_'):
    """
    Flattens a nested dict.
    Code taken from https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary/
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(convert_flatten(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def test(model, dataloader):
    model.network.eval()
    for input_batch, target_batch in dataloader:
        _ = model.evaluate(input_batch, target_batch)
    results = model.flush_val_metrics()
    results = convert_flatten(results)
    return results 

def main(model_path=None,  w=False):
    model = load_model(model_path)
    args = model.args
    args.train = False
    data = Dataset_handler(args)
    dataloader = data.get_loader(training=False)
    df = dataloader.dataset.table_data
    results = []
    args.test_fold = args.test_fold
    results = test(model, dataloader)
    results['test'] = '{}'.format(args.test_fold)
    if w:
        df_res = pd.DataFrame(results, index=[0])
        df_res.to_csv('results_test_{}.csv'.format(args.test_fold), index=False)
    return results 

