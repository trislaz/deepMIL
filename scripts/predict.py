from argparse import ArgumentParser
from deepmil.predict import load_model, predict_test, predict
import seaborn as sns
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, help='path to the model to use')
parser.add_argument("--error_table", help='path to the error table', default=None)
parser.add_argument("--data_path", type=str, help='path to the data on which to predict', default=None)
parser.add_argument("--data_table", help='path to the data table if provided', default=None)
parser.add_argument("--no_gt", action='store_true', help='no ground truth. No performance metrics is computed. Just prediction.')
args = parser.parse_args()


state = torch.load(args.model_path, map_location='cpu')
test = state['args'].test_fold
if args.no_gt:
    results_df = predict(model_path=args.model_path, data_path=args.data_path)
    results_df.to_csv('prediction.csv', index=False)
else:
    df, confusion_mat, target_correspondance = predict_test(model_path=args.model_path, 
            data_path=args.data_path, data_table=args.data_table)
    heatmap = sns.heatmap(confusion_mat, annot=True, cmap=plt.cm.Blues)
    heatmap.yaxis.set_ticklabels(target_correspondance, rotation=0, ha='right', fontsize=12)
    heatmap.xaxis.set_ticklabels(target_correspondance, rotation=45, ha='right', fontsize=12)
    plt.xlabel('predicted label')
    plt.ylabel('true label')
    
    ## For the datachallenge : computes the error with the error_table
    error = 0
    if args.error_table is not None:
        error_table = np.load(args.error_table)
        for i in range(confusion_mat.shape[0]):
            for j in range(confusion_mat.shape[1]):
                n = confusion_mat[i,j]
                err_i, err_j = target_correspondance[i], target_correspondance[j]
                error += error_table[err_i, err_j] * n
        error = error / confusion_mat.sum()
    
    model_name, _ = os.path.splitext(os.path.basename(args.model_path))
    root = os.path.dirname(args.model_path)
    confu_path = os.path.join(root, 'testfold_{}_error_{}.jpg'.format(test, (1-error)))
    csv_path= os.path.join(root, 'testfold_{}_error_{}.csv'.format(test, (1-error)))
    plt.savefig(confu_path)
    df.to_csv(csv_path, index=False)
