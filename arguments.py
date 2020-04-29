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
                        help='name of the model used. Avail : attentionmil')

    args = parser.parse_args()
    if args.patience == 0:
        args.patience = args.epochs
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

