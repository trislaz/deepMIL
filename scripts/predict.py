from argparse import ArgumentParser
from deepmil.predict import load_model
import seaborn as sns
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--config", type=str, help='path to the config file -.yaml format-')
parser.add_argument("--model_path", type=str, help='path to the model to use')
parser.add_argument("--out_name", type=str, help='name of the output file', default='prediction_output')
args = parser.parse_args()

df, confusion_mat, target_correspondance = load_model(config=args.config, model_path=args.model_path)
heatmap = sns.heatmap(confusion_mat, annot=True, cmap=plt.cm.Blues, normaliz)
heatmap.yaxis.set_ticklabels(target_correspondance, rotation=0, ha='right', fontsize=12)
heatmap.xaxis.set_ticklabels(target_correspondance, rotation=45, ha='right', fontsize=12)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig(args.out_name + '_confu_test.jpg')
df.to_csv(args.out_name+'.csv', index=False)
