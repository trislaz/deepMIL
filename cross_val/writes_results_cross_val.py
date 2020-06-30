"""Writes results from the output directory of the 
Train process in run.nf.
"""
from collections import MutableMapping 
from argparse import ArgumentParser
from glob import glob
import pandas as pd
import os
import torch
import shutil

def extract_config(config_path):
    """extracts the number of the config from its path (/../config_n.yaml)
    return int
    """
    config, _ = os.path.splitext(os.path.basename(config_path))
    config = int(config.split('_')[1])
    return config

def extract_references(args):
    """extracts a dictionnary with the parameters of a run
    return dict
    """
    t = args.test_fold
    r = args.repeat
    ref = {'test': t, 'repeat': r}
    return ref

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

def mean_dataframe(df):
    """Computes mean metrics for a set of repeat.
    for a given config c and a given test set t, computes
    1/r sum(metrics) over the repetitions.

    Parameters
    ----------
    df : pd.DataFrame
        dataframe of results. Columns contain config, test and repeat.
    
    returns:
    ----------
    df_mean_r = pd.DataFrame
        mean dataframe, w/o column repeat.
    df_mean_rt = pdf.DataFrame
        mean dataframe over repeats and then over test sets.
        mean of metrics for each config.
    """
    configs = set(df['config'])
    tests = set(df['test'])
    rows_r = []
    rows_rt = []
    for c in configs:
        dfc = df[df['config'] == c]
        rows_t = []
        for t in tests:
            dft = dfc[dfc['test'] == t]
            dft_m = dft.mean(axis=0)
            dft_m = dft_m.drop('repeat').to_frame().transpose()
            rows_r.append(dft_m)
            rows_t.append(dft_m)
        df_mean_t = pd.concat(rows_t, ignore_index=True)
        df_mean_t = df_mean_t.drop('test', axis=1)
        rows_rt.append(df_mean_t.mean(axis=0).to_frame().transpose())
    df_mean_r = pd.concat(rows_r, ignore_index=True)
    df_mean_rt = pd.concat(rows_rt, ignore_index=True)
    return df_mean_r, df_mean_rt

def select_best_repeat(df,ref_metric, path):
    """Selects, for a given config (best_config), the models
    that led to the best validation results = single run.
    Attention, best result is here the highest result. 
    (wont work when using the loss f.i)

    Parameters
    ----------
    df : pd.DataFrame
        dataframe with all results in it
    best_config : int
        already selected.
    ref_metric : str
        metric on which to select 
    path : str
        root path to the experiment.

    Returns
    -------
    list
        list containing tuples of parameters (config, test, repeat)
        that unequivocately leads to a model.
    """
    tests = set(df['test'])
    selection = []
    for t in tests:
        df_t = df[df['test'] == int(t)]
        best_rep = df.loc[df_t[ref_metric].idxmax(), 'repeat']
        selection.append((int(t), int(best_rep)))
    return selection

def copy_best_to_root(path, param):
    """Copy the best models for all the test_sets,
    and the config file in the root path of the experiment.
    if cross_val : just testing a single config. therefore no copy to do
    """
    t, r = param
    model_path = os.path.join(path, "test_{}/{}/model_best.pt.tar".format(t, r))
    shutil.copy(model_path, 'model_best_test_{}.pt.tar'.format(t))

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="folder where are stored the models.")
    args = parser.parse_args()
    models = glob(os.path.join(args.path, '**/*_best.pt.tar'), recursive=True)
    rows = [] 
    for m in models:
        state = torch.load(m, map_location='cpu')
        args_m = state['args']
        references = extract_references(args_m)
        metrics = state['best_metrics']
        #metrics = convert_flatten(metrics) I flattened the metrics directly in the models.py file
        references.update(metrics)
        rows.append(references)

    ref_metric = args_m.ref_metric # extract the reference from one of the models (last one)
    df = pd.DataFrame(rows)
    df_mean_r, df_mean_rt = mean_dataframe(df)
    models_params = select_best_repeat(df=df, best_config=best_config, ref_metric=ref_metric, path=args.path)
    for param in models_params:
        copy_best_to_root(args.path, param)
    df.to_csv('all_results.csv', index=False)
    df_mean_r.to_csv('mean_over_repeats.csv', index=False)
    df_mean_rt.to_csv('mean_over_tests.csv', index=False)

if __name__ == '__main__':
    main()
