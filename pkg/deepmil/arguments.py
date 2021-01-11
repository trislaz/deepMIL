from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import torch
import os
import copy
import yaml

def get_arguments(train=True, config=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',
                        type=str,
                        help='path to the config file. If None, use the command line parser',
                        default=config)
    # Data
    parser.add_argument("--wsi", type=str,help="path to the tiled WSI global folder (containing several resolutions)")
    # The structure of such a folder is usually $folder/size_256/res_$res/
    parser.add_argument("--target_name", type=str,help='name of the target variable.')
    parser.add_argument("--test_fold",type=int, help="number of the fold used as a test")
    parser.add_argument("--in_shape", type=int, default=32, help='size of the input tile, used if embedded=0')
    parser.add_argument('--sampler', type=str, help='type of tile sampler. dispo : random_sampler | random_biopsie', default='random_sampler')
    parser.add_argument('--val_sampler', type=str, help='sampler used for validation. Avail all | predmap_random | predmap_all', default='all')
    parser.add_argument('--resolution', type=int, help='level of resolution', default=2)
    # Data & model (in_layer)
    parser.add_argument('--embedded', type=int, default=1,
                        help='If 1, use the already embedded WSI, else, takes image_tiles as input')
    parser.add_argument('--features_net',  type=str, default='adm', 
                        help='type of feature extractor. Possible: adm | resnet ')
    parser.add_argument("--feature_depth", type=int, default=256, help="number of features to keep")
    parser.add_argument('--table_data', type=str, help='path to the csv containing the data info.')
    parser.add_argument('--model_name', type=str, default='attentionmil', help='name of the model used. Avail : attentionmil | 1s | transformermil | sa | conan | multiheadmil | multiheadmulticlass')
    parser.add_argument("--patience", type=int, default=0, help="Patience parameter for early stopping. If 0, then no early stopping is set (patience set to epochs)")
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size = how many WSI in a batch")
    parser.add_argument("--num_workers", type=int, default=8, help="number of parallel threads for batch processing")
    parser.add_argument('--nb_tiles',  type=int, default=0, help='number of tiles per WSI. If 0, the whole slide is processed.')
    parser.add_argument('--ref_metric', type=str, default='accuracy', help='reference metric for validation (which model to keep etc...')
    parser.add_argument('--repeat', type=int, default=1, help="identifier of the repetition. Used to ID the result.")
    parser.add_argument('--lr', type=float, help='learning rate', default=0.003)
    parser.add_argument('--dropout', type=float, help='dropout parameter', default=0)
    parser.add_argument('--color_aug',  type=int,help='If embedded = 0, will use color augmentation', default=1)
    parser.add_argument('--patience_lr',  type=int, help='number of epochs for the lr linear decay', default=None)
    parser.add_argument('--write_config', action='store_true', help='writes config in the cwd.')
    parser.add_argument('--atn_dim', type=int, help='intermediate projection dimension during attention mechanism. must be divisible by num_heads.', default=256)
    parser.add_argument('--num_heads', help='number of attention_head', default= 1, type=int)
    parser.add_argument('-k', type=int, help='number of selected tiles for the topk procedure (either when validating, or when training, with mhmc_conan)', default=5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_scheduler', type=str, default='linear')
    parser.add_argument('--criterion', type=str, help='criterion used', default=None)
    parser.add_argument('--error_table', type=str, help='path to the error table', default=None)

    parser.add_argument('--n_layers_classif', type=int, help='number of the internal layers of the classifier - works with model_name = mhmc_layers')

    if not train: # If test, nb_tiles = 0 (all tiles considered) and batc_size=1
        parser.add_argument("--model_path", type=str, help="Path to the model to load")
    args, _ = parser.parse_known_args()

    # If there is a config file, we populate args with it (still keeping the default arguments)
    if args.config is not None:
        with open(args.config, 'r') as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

    table = pd.read_csv(args.table_data)
    args.num_class = len(set(table[args.target_name]))
    args.wsi = os.path.join(args.wsi, 'res_{}'.format(args.resolution))
    # Arguments processing : either adding arguments with simple rules (constant size for example)
    # Or adding fixed arguments
    args.train = train
    if args.patience == 0:
        args.patience = args.epochs
    if args.features_net == 'resnet':
        args.feature_depth = 512
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.nb_tiles == 0:
        args.constant_size = False
    else:
        args.constant_size = True

    # Sgn_metric used to orient the early stopping and writing process.
    if args.ref_metric == 'loss':
        args.ref_metric = 'mean_val_loss'
        args.sgn_metric = 1
    else:
        args.sgn_metric = -1

    if args.model_name == 'multiheadmulticlass' and args.criterion is None:
        args.criterion = 'nll'

    if args.patience_lr is None:
        args.patience_lr = args.epochs
    
    dictio = copy.copy(vars(args))
    del dictio['device']
    config_str = yaml.dump(dictio)
    with open('./config.yaml', 'w') as config_file:
        config_file.write(config_str)

    if (args.criterion =='sosr') and (args.model_name == 'mhmc_layers'):
        print('Be careful, the output of mhmc_layers is logsoftmax, you need a linear output to use the \
                sosr criterion !')
      
    return args

