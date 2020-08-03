from sklearn.model_selection import StratifiedKFold
import pandas as pd
from argparse import ArgumentParser

# From a csv table, creates an other csv table with a "test" column, stating in which test fold are each line.
# The created Kfold is stratified with respect to $target_name, a column of the csv.

parser = ArgumentParser()
parser.add_argument('--table_in', type=str)
parser.add_argument('--table_out', type=str)
parser.add_argument('--target_name', type=str)
parser.add_argument('-k', type=int, help='number of folds')
args = parser.parse_args()

table = pd.read_csv(args.table_in)
target = args.target_name
output_path = args.table_out

skf = StratifiedKFold(n_splits=args.k, shuffle=True)
y = table[target].values
X = list(range(len(y)))
dic_test = dict()
for o, (train, test) in enumerate(skf.split(X, y)):
    for i in test:
        dic_test[i] = o
table["test"] = table.apply(lambda x: dic_test[x.name], axis=1) 
table.to_csv(args.table_out, index=False)






