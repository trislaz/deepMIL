from argparse import ArgumentParser
from deepmil.predict import load_model, predict
import seaborn as sns
import os
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument("--model_path", type=str, help='path to the model to use')
args = parser.parse_args()

df, confusion_mat, target_correspondance = predict(model_path=args.model_path)
heatmap = sns.heatmap(confusion_mat, annot=True, cmap=plt.cm.Blues)
heatmap.yaxis.set_ticklabels(target_correspondance, rotation=0, ha='right', fontsize=12)
heatmap.xaxis.set_ticklabels(target_correspondance, rotation=45, ha='right', fontsize=12)
plt.xlabel('predicted label')
plt.ylabel('true label')
model_name, _ = os.path.splitext(os.path.basename(args.model_path))
root = os.path.dirname(args.model_path)
confu_path = os.path.join(root, model_name+'_test_confu.jpg')
csv_path= os.path.join(root, model_name+'_test_res.csv')
plt.savefig(confu_path)
df.to_csv(csv_path, index=False)
