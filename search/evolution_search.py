import sys
# update your projecty root path before running
sys.path.insert(0, r'D:\Research\nsga-net')

import numpy as np
if not hasattr(np, "float"):   np.float   = float
if not hasattr(np, "int"):     np.int     = int
if not hasattr(np, "bool"):    np.bool    = bool
if not hasattr(np, "object"):  np.object  = object
if not hasattr(np, "complex"): np.complex = np.complex128

import os
import time
import logging
import argparse
from misc import utils

import numpy as np
from search import train_search
from search import micro_encoding
from search import macro_encoding
from search import nsganet as engine

from pymop.problem import Problem
from pymoo.optimize import minimize

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
# SynFlow arguments
parser.add_argument('--use_synflow', action='store_true', default=True, help='Use SynFlow for early stopping')
parser.add_argument('--no_synflow', action='store_true', help='Disable SynFlow early stopping')
args = parser.parse_args()
args.save = 'search_results/search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None, 
                 use_synflow=True, current_generation=0, max_generations=50):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0
        self._use_synflow = use_synflow
        self._current_generation = current_generation
        self._max_generations = max_generations

    def _evaluate(self, x, out, *args, **kwargs):
        from search.synflow import compute_population_synflow_scores, prefilter_architectures, rank_architectures
        from models.micro_models import NetworkCIFAR as Network
        from models.macro_models import EvoNetwork
        
        objs = np.full((x.shape[0], self.n_obj), np.nan)
        
        synflow_scores = []
        models = []
        
        if self._use_synflow:
            logging.info(f"Computing SynFlow scores for generation {self._current_generation}")
            
            for i in range(x.shape[0]):
                arch_id = self._n_evaluated + 1
                print('\n')
                logging.info('Computing SynFlow for Network id = {}'.format(arch_id))

                # Create model for SynFlow evaluation
                if self._search_space == 'micro':
                    genome = micro_encoding.convert(x[i, :])
                    genotype = micro_encoding.decode(genome)
                    model = Network(self._init_channels, 10, self._layers, False, genotype)
                elif self._search_space == 'macro':
                    genome = macro_encoding.convert(x[i, :])
                    genotype = macro_encoding.decode(genome)
                    channels = [(3, self._init_channels),
                              (self._init_channels, 2*self._init_channels),
                              (2*self._init_channels, 4*self._init_channels)]
                    model = EvoNetwork(genotype, channels, 10, (32, 32), decoder='residual')
                
                models.append(model)
                self._n_evaluated += 1
            
            # Compute SynFlow scores for all models
            synflow_scores = compute_population_synflow_scores(models)
            
            ranked_indices = rank_architectures(synflow_scores, return_indices=True)
            logging.info(f"SynFlow ranking for generation {self._current_generation}: {ranked_indices}")
            
            keep_ratio = 0.5
            keep_indices = prefilter_architectures(synflow_scores, len(synflow_scores), keep_ratio)
            logging.info(f"Keeping {len(keep_indices)} out of {len(synflow_scores)} architectures based on SynFlow prefiltering")
            
            for idx in keep_indices:
                arch_id = idx + 1
                logging.info(f"Training prefiltered architecture {arch_id} with SynFlow score {synflow_scores[idx]:.4f}")
                
                if self._search_space == 'micro':
                    genome = micro_encoding.convert(x[idx, :])
                elif self._search_space == 'macro':
                    genome = macro_encoding.convert(x[idx, :])
                
                performance = train_search.main(genome=genome,
                                                search_space=self._search_space,
                                                init_channels=self._init_channels,
                                                layers=self._layers, cutout=False,
                                                epochs=self._epochs,
                                                save='arch_{}'.format(arch_id),
                                                expr_root=self._save_dir,
                                                use_synflow=False,  # Disable early stopping
                                                synflow_threshold=0.0)

                objs[idx, 0] = 100 - performance['valid_acc']
                objs[idx, 1] = performance['flops']
            
            for idx in range(len(synflow_scores)):
                if idx not in keep_indices:
                    objs[idx, 0] = 100.0  # Poor accuracy
                    objs[idx, 1] = 1000.0  # High complexity penalty
                    logging.info(f"Architecture {idx+1} not selected for training (SynFlow score: {synflow_scores[idx]:.4f})")
        else:
            for i in range(x.shape[0]):
                arch_id = self._n_evaluated + 1
                print('\n')
                logging.info('Network id = {}'.format(arch_id))

                if self._search_space == 'micro':
                    genome = micro_encoding.convert(x[i, :])
                elif self._search_space == 'macro':
                    genome = macro_encoding.convert(x[i, :])
                    
                performance = train_search.main(genome=genome,
                                                search_space=self._search_space,
                                                init_channels=self._init_channels,
                                                layers=self._layers, cutout=False,
                                                epochs=self._epochs,
                                                save='arch_{}'.format(arch_id),
                                                expr_root=self._save_dir,
                                                use_synflow=False,
                                                synflow_threshold=0.0)

                objs[i, 0] = 100 - performance['valid_acc']
                objs[i, 1] = performance['flops']
                self._n_evaluated += 1

        out["F"] = objs


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    if hasattr(algorithm.problem, '_current_generation'):
        algorithm.problem._current_generation = gen

    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))


def main():
    np.random.seed(args.seed)
    logging.info("args = %s", args)

    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    else:
        raise NameError('Unknown search space type')

    use_synflow = args.use_synflow and not args.no_synflow
    
    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save,
                  use_synflow=use_synflow, max_generations=args.n_gens)

    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return


if __name__ == "__main__":
    main()