from argparse import ArgumentParser
from deepmil.predict import load_model

parser = ArgumentParser()
parser.add_argument("--config", type=str, help='path to the config file -.yaml format-')
parser.add_argument("--model_path", type=str, help='path to the model to use')
parser.add_argument("--out_name", type=str, help='name of the output file', default='prediction_output')
args = parser.parse_args()

df = load_model(config=args.config, model_path=args.model_path)
df.to_csv(args.out_name+'.csv', index=False)
