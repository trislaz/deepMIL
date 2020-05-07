from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

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
    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch Size = how many WSI in a batch")
    parser.add_argument("--patience", 
                        type=int,
                        default=0,
                        help="Patience parameter for early stopping. If 0, then no early stopping is set (patience set to epochs)")
    parser.add_argument('--epochs', 
                        type=int,
                        default=100,
                        help="number of epochs for training")
    parser.add_argument('--table_data', 
                        type=str, 
                        help='path to the csv containing the data info.')
    parser.add_argument('--model_name', 
                        type=str,
                        default='attentionmil', 
                        help='name of the model used. Avail : attentionmil | 1s')
    parser.add_argument('--nb_tiles',
                        type=int,
                        default=0,
                        help='number of tiles per WSI. If 0, the whole slide is processed.')

    args = parser.parse_args()

    # Arguments processing : either adding arguments with simple rules (constant size for example) 
    # Or adding fixed arguments
    if args.patience == 0:
        args.patience = args.epochs
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.nb_tiles == 0:
        args.constant_size = False
    else:
        args.constant_size = True

    args.dropout = 0
    return args

