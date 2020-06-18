from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
import yaml

def get_arguments(train=True):
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--config',
                        type=str,
                        help='path to the config file. If None, use the command line parser')
    # Data
    parser.add_argument("--path_tiles",
                        type=str, 
                        help="path to the tiled WSI",
                        required=True)
    parser.add_argument("--target_name",
                        type=str,
                        help='name of the target variable.')
    parser.add_argument("--test_fold", 
                        type=int,
                        help="number of the fold used as a test")
    # Data & model (in_layer)
    parser.add_argument("--feature_depth",
                        type=int,
                        default=2048,
                        help="number of features to keep")
    parser.add_argument('--table_data', 
                        type=str, 
                        help='path to the csv containing the data info.')
    parser.add_argument('--model_name', 
                        type=str,
                        default='attentionmil', 
                        help='name of the model used. Avail : attentionmil | 1s')
    if train:
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
    else: # If test, nb_tiles = 0 (all tiles considered) and batc_size=1
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
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.nb_tiles == 0:
        args.constant_size = False
    else:
        args.constant_size = True
    args.dropout = 0.3
    return args

