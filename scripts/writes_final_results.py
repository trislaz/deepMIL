import deepmil.test as test
from argparse import ArgumentParser
from glob import glob
import os
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', default='.', type=str, help='path to the folder where the best models are stored.')
    args = parser.parse_args()

    models_path = glob(os.path.join(args.path, 'model_best_test_*.pt.tar'), recursive=True)
    final_res = []
    for model in models_path:
        try:
            res = test.main(model_path=model)
        except:
            continue
        final_res.append(res)
    df_res = pd.DataFrame(final_res)
    mean = df_res.mean(axis=0).to_frame().transpose()
    std = df_res.std(axis=0).to_frame().transpose()
    mean['test'] = 'mean'
    std['test'] = 'std'
    df_res = pd.concat([df_res, mean, std])
    df_res = df_res.set_index('test')
    df_res.to_csv('final_results.csv')

if __name__ == '__main__':
    main()
