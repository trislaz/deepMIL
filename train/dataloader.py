from torch.utils.data import (Dataset, DataLoader, SubsetRandomSampler)
import pandas as pd
from PIL import Image
from glob import glob
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
import os

class SingleFolder(Dataset):
    imext = set(['.png', '.jpg'])
    def __init__(self, args, transform=None):
        super(SingleFolder, self).__init__()
        self.path = args.wsi
        self.transform = transform
        self.target_name = args.target_name
        self.n_sample = args.n_sample
        self.table_data = pd.read_csv(args.table_data)
        self.target_dict = dict()
        self.imext = set(['.png', '.jpg'])
        self.files = self._collect_files(args.wsi)
        
    def _collect_file_path(self, path):
        """Creates a list of all path to the images.
        """
        out = []
        slide = os.path.basename(path)
        _ = self.make_target_dict(slide)
        files = os.listdir(path)
        files = [os.path.splitext(x) for x in files]
        for name, ext in files:
            if ext in self.imext:
                out.append((os.path.join(path, name+ext), path))  
        return out 

    def make_target_dict(self, slide):
        """extracts targets of f from the table_data
        
        Parameters
        ----------
        slide : str
            name of the slide
        """
        is_in_table = slide in set(self.table_data['ID'])
        if is_in_table:
            target = self.table_data[self.table_data['ID'] == slide]['target'].values[0]
            self.target_dict[slide] = target
        return is_in_table
    
    def _collect_files(self, path):
        self.transform_target()
        return self._collect_file_path(path)

    def __len__(self):
        return len(self.files)

    def transform_target(self):
        """Adds to table to self.table_data
        a numerical encoding of the target. Works for classif.
        New columns is named "target"
        """
        table = self.table_data
        T = pd.factorize(table[self.target_name])
        table['target'] = T[0]
        self.target_correspondance = T[1]
        self.table_data = table

    def __getitem__(self, idx):
        impath, slide_path = self.files[idx]
        image = Image.open(impath)
        name_slide = os.path.basename(slide_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, float(self.target_dict[name_slide])

class FolderWSI(SingleFolder):
    def __init__(self, args, transform=None):
        """Instantiate a WSI-tiles dataset.
        args must contain :
            * target_name, str, name of the target
            * n_sample, int, number of tiles per WSI to sample
            * table_data, str, path to the csv containing data info (one column must be $target_name)
            * wsi, str, path to the data folder
        
        The datafolder (at args.wsi/) must be organized as follow:
            *$args.wsi/:
                * $ID_wsi1/: #ID_wsi1 being the values contained in table_data['ID'] identifying a WSI
                    * tile_1.jpg
                    * tile_2.jpg
                    [...]
                * $ID_wsi2/:
                    * tile_1.jpg
                    * tile_2.jpg
                    [...]
                [...]

        Parameters
        ----------
        args : Namespace
        transform : torchvision.transforms, optional
            transf to apply to the images, by default None
        """
        super(FolderWSI, self).__init__(args, transform)

    def _collect_files(self, path):
        """Collects all files : path is a folder full of folders, each being a wsi.
        """
        out = []
        self.transform_target()
        paths = glob(os.path.join(path, '*'))
        for p in paths:
            is_in_table = self.make_target_dict(os.path.basename(p))
            if is_in_table:
                all_patches = self._collect_file_path(p)
                np.random.shuffle(all_patches)
                out += all_patches[:self.n_sample] #Randomly choose the first samples 
        return out

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
                * nb_tiles, int, if 0 : take all the tiles, will need custom collate_fn, else randomly picks $nb_tiles in each WSI.
                * train, bool, if True : extract the data s.t fold != test_fold, if False s.t. fold == test_fold

        """
        super(EmbededWSI, self).__init__()
        self.args = args
        self.table_data = pd.read_csv(args.table_data)
        self.files, self.target_dict = self._make_db()
        self.constant_size = (args.nb_tiles != 0)
        
    def _make_db(self):
        table = self.transform_target()
        target_dict = dict() #Key = path to the file, value=target
        files = glob(os.path.join(self.args.path_tiles, '*.npy'))
        files_filtered =[]
        for f in files:
            name, _ = os.path.splitext(os.path.basename(f))
            if self._is_in_db(f):
                files_filtered.append(f)
                target_dict[f] = np.float32(table[table['ID'] == name]['target'].values[0])
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
        is_in_db = name in table['ID'].values
        if 'test' in table.columns:
            is_in_train = (table[table['ID'] == name]['test'] != self.args.test_fold).item() if is_in_db else False # "keep if i'm not test"
            is_in_test = (table[table['ID'] == name]['test'] == self.args.test_fold).item() if is_in_db else False
            is_in_db = is_in_train if args.train else is_in_test
        return is_in_db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mat = np.load(path)[:,:self.args.feature_depth]
        mat = self._select_tiles(mat)
        mat = torch.from_numpy(mat).float() #ToTensor
        target = self.target_dict[path]
        return mat, target

    def _select_tiles(self, mat):
        """Select the tiles from a WSI.
        If we want a constant size, then $nb_tiles are randomly draw from the available tiles.
        If not, return the whole matrix.

        Parameters
        ----------
        mat : np.array
            matrix of the encoded WSI.

        Returns
        -------
        np.array
            selected tiles 
        """
        if self.constant_size:
            indexes = np.random.randint(mat.shape[0], size=self.args.nb_tiles)
            mat = mat[indexes, :]
        return mat

def collate_variable_size(batch):
    data = [item[0].unsqueeze(0) for item in batch]
    target = [torch.FloatTensor([item[1]]) for item in batch]
    return [data, target]

def make_loaders(args):
    dataset = EmbededWSI(args=args)
    if args.train: # In a context of cross validation.
        labels = [x[1] for x in dataset]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

        # Shuffles dataset
        train_indices, val_indices = [x for x in splitter.split(X=labels, y=labels)][0]
        val_sampler = SubsetRandomSampler(val_indices)
        train_sampler = SubsetRandomSampler(train_indices)

        # Collating regime
        if args.constant_size:
            collate = None
        else:
            collate = collate_variable_size

        dataloader_train = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=24, collate_fn=collate)
        dataloader_val = DataLoader(dataset=dataset, batch_size=1, sampler=val_sampler, num_workers=24)
        dataloaders = (dataloader_train, dataloader_val)
    else: # Testing on the whole dataset
        dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=24)
        dataloaders = (dataloader)
    return dataloaders



if __name__ == '__main__':
    ## To test it
    from argparse import Namespace
    args = Namespace()
    args.table_data = "/Users/trislaz/Documents/cbio/projets/deepMIL_tris/labels_tcga_tnbc_strat.csv"
    args.path_tiles = "/Users/trislaz/Documents/cbio/data/tcga/TCGA_TNBC/encoded/imagenet_R_2/2/mat_pca"
    args.target_name = "LST_status"
    args.batch_size = 2
    args.test_fold = 1
    args.nb_tiles = 10
    args.feature_depth = 2048
    args.device = torch.device('cpu')

    db = EmbededWSI(args)
    train, val = make_loaders(args)
    for x, y in train:
        print(x)
        print(y)
