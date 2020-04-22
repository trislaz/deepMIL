from torch.utils.data import (Dataset, DataLoader, SubsetRandomSampler)
import pandas as pd
from glob import glob
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import os

class EmbededWSI(Dataset):
    """
    DO NOT PRELOAD DATASET ON RAM. may be slow.
    Implements a dataloader for already coded WSI.
    Each WSI is therefore a .npy array of size NxF with N the number of tiles 
    of the WSI and F the number of features of the embeding space (usually 2048).
    Note: no transform method because this dataset is using numpy array as inputs.

    The table_data (labels of the different files) may have:
        * an ID column with the name of the images in it (without the extension)
        * a $args.target_name column, of course
        * a test columns, stating the test_fold number of each image.
    """
    def __init__(self, args):
        """Initialises the MIL model.
        
        Parameters
        ----------
        args : Namespace
            must contain :
                * table_data, str, path to the data info (.csv), with 'ID' column containing name of wsi.
                * path_tiles, str, path to the embedded WSI (.npy) with name matching the 'ID' of table_data
                * target_name, str, name of the target variable (name of column in table_data)
                * device, torch.device
                * test_fold, int, number of the fold used as test.
                * feature_depth, int, number of dimension of the embedded space to keep. (0<x<2048)

        """
        super(EmbededWSI, self).__init__()
        self.args = args
        self.table_data = pd.read_csv(args.table_data)
        self.files, self.target_dict = self._make_db()
        
    def _make_db(self):
        table = self.transform_target()
        target_dict = dict() #Key = path to the file, value=target
        files = glob(os.path.join(self.args.path_tiles, '*.npy'))
        files_filtered =[]
        for f in files:
            name, _ = os.path.splitext(os.path.basename(f))
            if self._is_in_db(f):
                files_filtered.append(f)
                target_dict[f] = table[table['ID'] == name]['target'].values[0]
        return files_filtered, target_dict

    def transform_target(self):
        """Adds to table a numerical encoding of the target.
        Each class is a natural number. Good format for classif using nn.CrossEntropy
        New columns is named "target"
        """
        table = self.table_data
        T = pd.factorize(table[self.args.target_name])
        table['target'] = T[0]
        self.target_correspondance = T[1]
        self.table_data = table
        return table

    def _is_in_db(self, f):
        """Do we keep the file in the dataset ?
        To test : 
            * Is the file in the table_data ?
            * Is the file in the test set ?
            * Is'nt the file an outsider .. ?
            * Other reason to exclude an image.
        """
        table = self.table_data
        name, _ = os.path.splitext(os.path.basename(f))
        is_in_train = (table[table['ID'] == name]['test'] != self.args.test_fold).item() # "keep if i'm not test"
        is_in_db = is_in_train # & is_not_forbidden & is_in_table
        return is_in_db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mat = np.load(path)[:,:self.args.feature_depth]
        mat = torch.from_numpy(mat).float() #ToTensor
        target = self.target_dict[path]
        return mat, target

def make_loaders(args):
    dataset = EmbededWSI(args=args)
    labels = [x[1] for x in dataset]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    # Shuffles dataset
    train_indices, val_indices = splitter.split(X = labels, y=labels)
    val_sampler = SubsetRandomSampler(val_indices)
    train_sampler = SubsetRandomSampler(train_indices)

    dataloader_train = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=24)
    dataloader_val = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=24)
    return dataloader_train, dataloader_val


if __name__ == '__main__':
    ## To test it
    from argparse import Namespace
    args = Namespace()
    args.table_data = "/Users/trislaz/Documents/cbio/projets/deepMIL_tris/minimal_example/labels_luminaux_test.csv"
    args.path_tiles = "/Users/trislaz/Documents/cbio/projets/deepMIL_tris/minimal_example/numpy_WSI"
    args.target_name = "HRD"
    args.test_fold = 1
    args.feature_depth = 2048
    args.device = torch.device('cpu')

    db = EmbededWSI(args)

    db[0]

