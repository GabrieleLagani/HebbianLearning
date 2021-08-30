import os
import argparse
import scipy.stats as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from .. import params as P
from . import utils


# This is a utility script that takes a path to a desired result folder, reads the results saved in this
# folder, and then plots mean and confidence intervals of training series.
def run_dispconv(path, label, crit, seeds, ci_levels, savepath, fillbetw):
	series = [{} for _ in range(len(path))]
	series_avg = []
	series_se = []
	for i in range(len(path)):
		for id in seeds:
			checkpoint_folder = os.path.join(path[i], 'iter' + str(id), 'checkpoints')
			checkpoint_list = utils.get_checkpoint_list(checkpoint_folder)
			checkpoint_id = max(checkpoint_list)
			checkpoint_file_path = os.path.join(checkpoint_folder, "checkpoint" + str(checkpoint_id) + ".pt")
			loaded_checkpoint = utils.load_dict(checkpoint_file_path)
			series[i][id] = list(loaded_checkpoint['val_result_data'][loaded_checkpoint['crit_names'].index(crit)].values())
		series_avg.append(np.mean([series[id] for id in seeds], axis=0).tolist())
		series_se.append(st.sem([series[id] for id in seeds], axis=0).tolist())
	for ci_lvl in ci_levels:
		graph = plt.axes(xlabel='epoch', ylabel=crit)
		for i in range(len(path)):
			ci = st.t.interval(ci_lvl, len(seeds) - 1, loc=series_avg[i], scale=series_se[i])
			if not fillbetw:
				graph.errorbar(list(series[i][0].keys()), series_avg[i], yerr=((ci[1] - ci[0])/2).tolist(), fmt='o', capsize=3, label=label[i])
			else:
				graph.plot(list(series[i][0].keys()), series_avg[i], label=label[i])
				graph.fill_between(list(series[i][0].keys()), ci[0], ci[1], alpha=.1)
		graph.grid(True)
		graph.legend()
		os.makedirs(os.path.dirname(savepath), exist_ok=True)
		fig = graph.get_figure()
		fig.savefig(savepath + '_ci_' + str(ci_lvl) + '.png', bbox_inches='tight')
		plt.close(fig)
			

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', nargs='*', default=P.DEFAULT_DISPSERIES_PATHS, type=str, help="Path to the csv file you want to examine.")
	parser.add_argument('--label', nargs='*', default=P.DEFAULT_DISPSERIES_LABELS, type=str, help="Name to be given to the each series.")
	parser.add_argument('--crit', default=P.DEFAULT_CRIT, help="Name of criterion you want to evaluate.")
	parser.add_argument('--seeds', nargs='*', default=P.DEFAULT_SEEDS, type=int, help="The RNG seeds of the experiments you want to examine.")
	parser.add_argument('--ci', nargs='*', default=P.DEFAULT_CI_LEVELS, type=float, help="Confidence interval levels you want to report.")
	parser.add_argument('--savepath', default=P.DEFAULT_DISPSERIES_SAVEPATH, type=str, help="The path (without extension) where to save the resulting plots.")
	parser.add_argument('--fillbetw', action='store_true', default=P.DEFAULT_DISPSERIES_FILLBETW, help="Using this flag, confidence intervals will be displayed as fill between mode, otherwise they will be displayed as bars.")
	args = parser.parse_args()
	
	run_dispconv(path=args.path, label=args.label, crit=args.crit, seeds=args.seeds, ci_levels=args.ci, savepath=args.savepath, fillbetw=args.fillbetw)