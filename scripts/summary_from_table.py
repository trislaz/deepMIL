from argparse import ArgumentParser
from summary_fig import main
from deepmil.predict import get_args
import os
import pandas as pd

parser = ArgumentParser()
parser.add_argument('--model_path', type=str, help='path to the model used for prediction')
parser.add_argument('--out', type=str, help='name of the folder where storing the outputs', default='summaries')
parser.add_argument('--with_gt', type=int, help='If 1, then the gt can be found in table, else 0.', default=1)
args = parser.parse_args()
args = get_args(args)
table = args.table if args.with_gt else None
valid_ID = set([os.path.splitext(os.path.basename(x))[0] for x in os.listdir(args.raw_path)])
df = pd.read_csv(args.table)
IDs = df['ID'].values
for i in IDs:
    if i in valid_ID:
        main(model_path=args.model_path, wsi_ID=i, embed_path=args.embed_path,
            raw_path=args.raw_path, out=args.out, table=table)
