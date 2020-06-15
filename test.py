"""
Test a given model.
"""
import numpy as 
import pandas as pd
from torch import load
from arguments import get_arguments
from models import DeepMIL
from dataloader import make_loaders

def test(model, dataloader):
    model.network.eval()
    for input_batch, target_batch in dataloader:
        _ = model.evaluate(input_batch, target_batch)
    results = model.flush_val_metrics()
    df_res = pd.DataFrame.from_dict(results)
    return df_res

def main():
    args = get_arguments(train=False)
    model = DeepMIL(args=args)
    state = torch.load(args.model_path)
    model.load_state_dict(state['state_dict'])
    dataloader = make_loaders(args)
    df_res = test(model, dataloader)
    df_res.to_csv('results_test_{}.csv'.format(args.test_fold))