"""
Test a given model.
"""
import numpy as np
import pandas as pd
import os
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

def test(model, dataloader, table):
    model.network.eval()
    for input_batch, target_batch in dataloader:
        _ = model.evaluate(input_batch, target_batch)
    proba_preds = model.results_val['proba_preds']
    preds = model.results_val['preds']
    ids = [os.path.splitext(os.path.basename(x))[0] for x in dataloader.dataset.files]
    result_table = fill_table(table, proba_preds, preds, ids)
    results = model.flush_val_metrics()
    results = convert_flatten(results)
    return results, result_table

def fill_table(table, proba_preds, preds, ids):
    """
    returns the "data_table" with the additional columsn scores and preds.
    scores, preds, ids are lists, indices correspond to the same image.
    """
    pi_scores = []
    pi_preds = []
    def is_in_set(x):
        if x['ID'] in ids:
            return True
        else: 
            return False
    table['take'] = table.apply(is_in_set, axis=1)
    table = table[table['take']]
    for i in table['ID'].values:
        index = ids.index(i)
        pi_scores.append(proba_preds[index])
        pi_preds.append(preds[index])
    table['proba_preds'] = pi_scores
    table['preds'] = pi_preds
    return table

def main(model_path=None,  w=False, rm_duplicates=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    args = model.args
    table = pd.read_csv(args.table_data)
    if rm_duplicates: # Allows the use of upsampling.
        table = table.drop_duplicates()
    args.table_data = table
    args.train = False
    data = Dataset_handler(args)
    dataloader = data.get_loader(training=False)
    df = dataloader.dataset.table_data
    results = []
    args.test_fold = args.test_fold
    results, result_table = test(model, dataloader, table)
    results['test'] = '{}'.format(args.test_fold)
    if w:
        df_res = pd.DataFrame(results, index=[0])
        df_res.to_csv('results_test_{}.csv'.format(args.test_fold), index=False)
    return results, result_table

