from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import yaml

def get_arguments(train=True, config=None):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',
                        type=str,
                        help='path to the config file. If None, use the command line parser', 
                        default=config)
    # Data
    parser.add_argument("--wsi",
                        type=str, 
                        help="path to the tiled WSI")
    parser.add_argument("--target_name",
                        type=str,
                        help='name of the target variable.')
    parser.add_argument("--test_fold", 
                        type=int,
                        help="number of the fold used as a test")
    parser.add_argument("--in_shape", 
                        type=int,
                        default=32,
                        help='size of the input tile, used if embedded=0')
    # Data & model (in_layer)
    parser.add_argument('--embedded', 
                        type=int,
                        default=1,
                        help='If 1, use the already embedded WSI, else, takes image_tiles as input')
    parser.add_argument('--features_net', 
                        type=str, 
                        default='adm', 
                        help='type of feature extractor. Possible: adm | resnet ')
    parser.add_argument("--feature_depth",
                        type=int,
                        default=256,
                        help="number of features to keep")
    parser.add_argument('--table_data', 
                        type=str, 
                        help='path to the csv containing the data info.')
    parser.add_argument('--model_name', 
                        type=str,
                        default='attentionmil', 
                        help='name of the model used. Avail : attentionmil | 1s')
    parser.add_argument("--patience", 
                        type=int,
                        default=0,
                        help="Patience parameter for early stopping. If 0, then no early stopping is set (patience set to epochs)")
    parser.add_argument('--epochs', 
                        type=int,
                        default=100,
                        help="number of epochs for training")
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch Size = how many WSI in a batch")
    parser.add_argument("--num_workers",
                        type=int,
                        default=16,
                        help="number of parallel threads for batch processing")
    parser.add_argument('--nb_tiles',
                        type=int,
                        default=0,
                        help='number of tiles per WSI. If 0, the whole slide is processed.')
    parser.add_argument('--ref_metric',
                        type=str,
                        default='accuracy',
                        help='reference metric for validation (which model to keep etc...')
    parser.add_argument('--repeat',
                        type=int,
                        default=1,
                        help="identifier of the repetition. Used to ID the result.")
    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=0.003)
    parser.add_argument('--dropout',
                        type=float,
                        help='dropout parameter',
                        default=0)
    parser.add_argument('--color_aug', 
                        type=int,
                        help='If embedded = 0, will use color augmentation',
                        default=1)
    if not train: # If test, nb_tiles = 0 (all tiles considered) and batc_size=1
        parser.add_argument("--model_path",
                            type=str,
                            help="Path to the model to load")
    args, _ = parser.parse_known_args()

    # If there is a config file, we populate args with it (still keeping the default arguments)
    if args.config is not None:
        with open(args.config, 'r') as f:
            dic = yaml.safe_load(f)
        args.__dict__.update(dic)

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
    return args

