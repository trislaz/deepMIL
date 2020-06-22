"""
Samples hyperparameters to feed the pipeline.
Outputs :
    * yaml config file, giving the whole parameters set.
    * an empty results file (python .pickle that will contain results for each run)
"""
import yaml
import numpy as np
from argparse import ArgumentParser

class Sampler:
    parameters_per_model = {"attentionmil": ['width_fe'],
                        "1s": ['n_clusters', 'hidden_fcn'],
                        "conan":['hidden1', 'hidden2', 'hidden_fcn']}
    global_parameters = ["lr", "batch_size", "nb_tiles", "dropout"]
    res_to_tiles = {1: (100, 4000), 2: (10, 400)} #donne la fourchette du nb de tuiles pour chaque rés

    def __init__(self, args):
        self.res = args.res
        self.params = vars(args)
        self.table_data = args.table_data
        self.path_tiles = args.path_tiles
        self.model_name = args.model_name
        self.p = 0.2 # in nb_tiles
        self.parameters_name = self.parameters_per_model[args.model_name] + self.global_parameters
        #self.sampling_function = {"lr": self.lr_sampler,
        #                "batch_size": self.batch_size_sampler,
        #                "nb_tiles": self.nb_tiles_sampler,
        #                "dropout": self.dropout_sampler}

    @staticmethod
    def lr_sampler(high=1, low=3):
        """Log samples the learning rate, in a
        window between 10^-high and 10^-4
        """
        s = np.random.uniform(high, low) # s for sampler
        s = 10 ** (-s)
        return s

    @staticmethod
    def width_fe_sampler():
        s = np.random.choice([16, 32, 64, 128, 256])
        return s

   @staticmethod 
    def n_clusters_sampler():
        s = np.random.choice([16, 32, 64, 128, 256])
        return s

    @staticmethod    
    def hidden_fcn_sampler():
        s = np.random.choice([[16, 32, 64, 128, 256]])
        return s
        
    @staticmethod
    def hidden1_sampler():
        s = np.random.choice([[16, 32, 64, 128, 256]])
        return s
    
    @staticmethod
    def batch_size_sampler():
        """Uniformly samples the batch_size
        """
        s = np.random.randint(1, 16)
        return s

    @staticmethod
    def dropout_sampler():
        d = np.random.uniform(0, 0.5)
        return d

    def nb_tiles_sampler(self):
        low, high = self.res_to_tiles[self.res]
        take_all_tiles = np.random.binomial(1, self.p)
        if take_all_tiles:
            s = 0
        else:
            s = np.random.randint(low, high)
        return s

    def sample(self):
        for p in self.parameters_name:
            sampling_function = getattr(self, p+'_sampler')
            self.params[p] = sampling_function()
        return self.params

def main():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--path_tiles", help="path to the encoded wsi")
    parser.add_argument("--table_data", help='path to the table data')
    parser.add_argument("--id", help="ID of the config", type=int)
    parser.add_argument("--res", type=int, help="resolution of the wsi")
    parser.add_argument("--target_name", type=str, help='name of the target in the table_data')
    args = parser.parse_args()
    sampler = Sampler(args)
    params = sampler.sample()
    with open('config_{}.yaml'.format(args.id), 'w') as config:
        config.write(yaml.dump(params))

if __name__ == "__main__":
    main()        
