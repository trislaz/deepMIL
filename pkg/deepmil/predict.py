"""Just predicts, given a set of WSI.
"""
import numpy as np
import seaborn as sns
from sklearn import metrics
import pandas as pd
from glob import glob
import torch
import pandas as pd
from torch import load
import os
from .arguments import get_arguments
from .models import DeepMIL
from .dataloader import EmbededWSI, Dataset_handler
from collections import MutableMapping

def load_model(model_path, device):
    """Loads and prepare a learned model for prediction.

    Args:
        model_path (str): path to the *.pt.tar model
    """
    checkpoint = torch.load(model_path, map_location='cpu')
    args = checkpoint['args']
    args.target_correspondance = checkpoint['dataset'].target_correspondance
    args.device = device
    args.optimizer, args.lr_scheduler = 'adam', 'cos'
    #args.model_name = 'mhmc_conan' # Try conan at the test state
    #args.k = 5
    model = DeepMIL(args)
    model.network.load_state_dict(checkpoint['state_dict'])
    model.target_correspondance = checkpoint['dataset'].target_correspondance
    model.network.eval()
    return model

def predict_test(model_path=None, data_path=None,data_table=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    args = model.args
    if data_path is not None:
        args.wsi = data_path
        if data_table is not None:
            args.table_data = data_table
        else:
            print("careful, you will try to get performance prediction on a new dataset, with the training labels")
    data = Dataset_handler(model.args, predict=True)
    dataloader = data.get_loader(training=False)
    df = dataloader.dataset.table_data
    results = []
    model.network.eval()
    for o, (x, y) in enumerate(dataloader):
        print(x.shape)
        proba, y_hat = model.predict(x)
        id_im = os.path.splitext(os.path.basename(dataloader.dataset.files[o]))[0].split('_embedded')[0]
        serie = df[df['ID'] == id_im].to_dict('records')[0]
        success = y_hat.item() == model.target_correspondance[y.item()]
        r = {'prediction': y_hat.item(), 'gt': model.target_correspondance[y.item()], 'index':o,'success': success, 'pp0': proba[0][0], 'pp1': proba[0][1],'pp2': proba[0][2], 'pp3': proba[0][3]} # pp pour pseudo_proba
        r.update(serie) 
        results.append(r)
    results = pd.DataFrame(results)
    results_test = results[results['test'] == model.args.test_fold]
    predicted_labels = results_test['prediction'].values
    true_labels = results_test['gt'].values
    confusion_mat = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    return results, confusion_mat, dataloader.dataset.target_correspondance 

def predict(model_path, data_path):
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, device)
    wsis = glob(os.path.join(data_path, '*.npy'))
    assert wsis, "The images have to be stored directly into data_path, with a npy extension.x:  data_path/slide_1.npy"
    for wsi in wsis:
        name = os.path.basename(wsi).replace('.npy', '.tif')
        wsi = preprocessing(wsi, device)
        proba, y_hat = model.predict(wsi)
        results.append({'filename': name, 'pred': y_hat.item(), 'pp0': proba[0][0], 'pp1': proba[0][1],'pp2': proba[0][2], 'pp3': proba[0][3]})
    results_df = pd.DataFrame(results)
    return results_df

def preprocessing(wsi, device):
    wsi = np.load(wsi)
    wsi = torch.Tensor(wsi)
    wsi = wsi.unsqueeze(0)
    wsi = wsi.to(device).float()
    return wsi
