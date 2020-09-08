from torch.utils.data import (Dataset, DataLoader, SubsetRandomSampler)
import pandas as pd
from PIL import Image
from glob import glob
import numpy as np
import torch
from tiler_wsi.tile_retriever.tile_sampler import TileSampler
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import os

class FolderWSI(Dataset):
    def __init__(self, args, train, transform=None, predict=False):
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
        super(FolderWSI, self).__init__()
        self.args = args
        self.train = train
        self.table_data = pd.read_csv(args.table_data)
        self.predict = predict
        self.target_name = args.target_name
        self._transform_target()
        self.images = dict()
        self.target_dict = dict()
        self._get_images(args.wsi)
        self.transform = transform
        self.nb_tiles = args.nb_tiles
        
    def _get_images(self, path):
        possible_dir = os.listdir(path)
        for d in possible_dir:
            if os.path.isdir(os.path.join(path, d)):
                if self._make_target_dict(d):
                    self.images[d] = glob(os.path.join(path, d, '*.jpg'))

    def _transform_target(self):
        """Adds to table to self.table_data
        a numerical encoding of the target. Works for classif.
        New columns is named "target"
        """
        table = self.table_data
        T = pd.factorize(table[self.target_name])
        table['target'] = T[0]
        self.target_correspondance = T[1]
        self.table_data = table    

    def _make_target_dict(self, slide):
        """extracts targets of f from the table_data
        
        Parameters
        ----------
        slide : str
            name of the slide
        """
        is_in_db = self._is_in_db(slide)
        if is_in_db:
            target = self.table_data[self.table_data['ID'] == slide]['target'].values[0]
            self.target_dict[slide] = target
        return is_in_db

    def _is_in_db(self, slide):
        """Do we keep the file in the dataset ?
        To test :
            * Is the file in the table_data ?
            * Is the file in the test set ?
            * Is'nt the file an outsider .. ?
            * Other reason to exclude an image.
        """
        table = self.table_data
        is_in_db = slide in set(table['ID'])
        if 'test' in table.columns and (not self.predict):
            is_in_train = (table[table['ID'] == slide]['test'] != self.args.test_fold).values[0] if is_in_db else False # "keep if i'm not test"
            is_in_test = (table[table['ID'] == slide]['test'] == self.args.test_fold).values[0] if is_in_db else False
            is_in_db = is_in_train if self.args.train else is_in_test
        return is_in_db

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        wsi = list(self.images)[idx]
        impath = self.images[wsi]
        np.random.shuffle(impath)
        instances = []
        if self.train:
            nb_tiles = self.nb_tiles
        else:
            nb_tiles = min(280, len(impath))
        for i in range(self.nb_tiles):
            instance = Image.open(impath[i])
            if self.transform is not None:
                instance = self.transform(instance)
            instances.append(instance)
        instances = torch.stack(instances)
        instances = instances.permute(1, 0, 2, 3)
        return instances, self.target_dict[wsi]

class EmbededWSI(Dataset):
    """
    DO NOT PRELOAD DATASET ON RAM. may be slow.
    OTHER SOLUTION THAT MAY WORK FASTER : write each tile as a different file. Then load each of them and 
    concatenate them to create a WSI.
    Implements a dataloader for already coded WSI.
    Each WSI is therefore a .npy array of size NxF with N the number of tiles 
    of the WSI and F the number of features of the embeding space (usually 2048).
    Note: no transform method because this dataset is using numpy array as inputs.

    The table_data (labels of the different files) may have:
        * an ID column with the name of the images in it (without the extension)
        * a $args.target_name column, of course
        * a test columns, stating the test_fold number of each image.
    """
    def __init__(self, args, train, predict=False):
        """Initialises the MIL model.
        
        Parameters
        ----------
        args : Namespace
            must contain :
                * table_data, str, path to the data info (.csv), with 'ID' column containing name of wsi.
                * wsi, str, path to the the output folder of a tile_image process. #embedded WSI (.npy) with name matching the 'ID' of table_data
                * target_name, str, name of the target variable (name of column in table_data)
                * device, torch.device
                * test_fold, int, number of the fold used as test.
                * feature_depth, int, number of dimension of the embedded space to keep. (0<x<2048)
                * nb_tiles, int, if 0 : take all the tiles, will need custom collate_fn, else randomly picks $nb_tiles in each WSI.
                * train, bool, if True : extract the data s.t fold != test_fold, if False s.t. fold == test_fold
                * sampler, str: tile sampler. dispo : random_sampler | random_biopsie

        """
        super(EmbededWSI, self).__init__()
        self.args = args
        self.embeddings = os.path.join(args.wsi, 'mat_pca')
        self.info = os.path.join(args.wsi, 'info')
        self.train = train
        self.predict = predict
        self.table_data = pd.read_csv(args.table_data)
        self.files, self.target_dict, self.sampler_dict = self._make_db()
        self.constant_size = (args.nb_tiles != 0)
        
    def _make_db(self):
        table = self.transform_target()
        target_dict = dict() #Key = path to the file, value=target
        sampler_dict = dict()
        files = glob(os.path.join(self.embeddings, '*.npy'))
        files_filtered =[]
        for f in files:
            name, _ = os.path.splitext(os.path.basename(f))
            name = name.split('_embedded')[0] # vire le "embedded"... dirty
            if self._is_in_db(name):
                files_filtered.append(f)
                target_dict[f] = np.float32(table[table['ID'] == name]['target'].values[0])
                sampler_dict[f] = TileSampler(wsi_path=f, info_folder=self.info)
        return files_filtered, target_dict, sampler_dict

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

    def _is_in_db(self, name):
        """Do we keep the file in the dataset ?
        To test :
            * Is the file in the table_data ?
            * Is the file in the test set ?
            * Is'nt the file an outsider .. ?
            * Other reason to exclude an image.
        """
        table = self.table_data
        is_in_db = name in table['ID'].values
        if 'test' in table.columns and (not self.predict):
            is_in_train = (table[table['ID'] == name]['test'] != self.args.test_fold).values[0] if is_in_db else False # "keep if i'm not test"
            is_in_test = (table[table['ID'] == name]['test'] == self.args.test_fold).values[0] if is_in_db else False
            is_in_db = is_in_train if self.args.train else is_in_test
        return is_in_db

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mat = np.load(path)[:,:self.args.feature_depth]
        mat = self._select_tiles(path, mat)
        mat = torch.from_numpy(mat).float() #ToTensor
        target = self.target_dict[path]
        return mat, target

    def _select_tiles(self, path, mat):
        """Select the tiles from a WSI.
        If we want a constant size, then $nb_tiles are randomly draw from the available tiles.
        If not, return the whole matrix.

        Parameters
        ----------
        path: str
            path of the WSI.

        mat : np.array
            matrix of the encoded WSI.

        Returns
        -------
        np.array
            selected tiles 
        """
        if self.train:
            sampler = self.sampler_dict[path]
            indices = getattr(sampler, self.args.sampler)(nb_tiles=self.args.nb_tiles)
            mat = mat[indices, :]
        return mat

def collate_variable_size(batch):
    data = [item[0].unsqueeze(0) for item in batch]
    target = [torch.FloatTensor([item[1]]) for item in batch]
    return [data, target]

class Dataset_handler:
    def __init__(self, args, predict=False):
        self.args = args
        self.predict = predict
        self.num_workers = args.num_workers 
        self.embedded = args.embedded
        self.dataset_train = self._get_dataset(train=True)
        self.dataset_val = self._get_dataset(train=False)
        self.train_sampler, self.val_sampler = self._get_sampler(self.dataset_train)

    def get_loader(self, training):
        if training:
            collate = None if self.args.constant_size else collate_variable_size
            dataloader_train = DataLoader(dataset=self.dataset_train, batch_size=self.args.batch_size, sampler=self.train_sampler, num_workers=self.num_workers, collate_fn=collate, drop_last=True)
            dataloader_val = DataLoader(dataset=self.dataset_val, batch_size=1, sampler=self.val_sampler, num_workers=self.num_workers)
            dataloaders = (dataloader_train, dataloader_val) 
        else: # Testing on the whole dataset
            dataloaders = DataLoader(dataset=self.dataset_val, batch_size=1, num_workers=self.num_workers)
        return dataloaders
        
    def _get_dataset(self, train):
        if self.embedded:
            dataset = EmbededWSI(self.args, train=train, predict=self.predict)
        else:
            dataset = FolderWSI(self.args, train=train, transform=get_transform(train=train, color_aug=self.args.color_aug))
        return dataset
    
    def _get_sampler(self, dataset):
        labels = [x[1] for x in dataset]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
        train_indices, val_indices = [x for x in splitter.split(X=labels, y=labels)][0]
        val_sampler = SubsetRandomSampler(val_indices)
        train_sampler = SubsetRandomSampler(train_indices)
        return train_sampler, val_sampler

def get_transform(train, color_aug=False):
    if train:
        if color_aug:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3)], p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform

if __name__ == '__main__':
    ## To test it
    from argparse import Namespace
    args = Namespace()
    args.table_data = "/Users/trislaz/Documents/cbio/data/breast_benchmark/bc_benchmark.csv"
    args.wsi = "/Users/trislaz/Documents/cbio/data/breast_benchmark/tiled/"
    args.target_name = "status"
    args.embedded = False
    args.batch_size = 2
    args.test_fold = 1
    args.color_aug = True
    args.constant_size = True
    args.nb_tiles = 10
    args.feature_depth = 256 
    args.device = torch.device('cpu')

    ob = Dataset_handler(args)
    train_l, val_l = ob.get_loader(True)
    db = FolderWSI(args, transform=get_transform(True, True))
    db2 = FolderWSI(args, transform=get_transform(True, True))
    for x, y in train_l:
        print(x)
        print(y)
