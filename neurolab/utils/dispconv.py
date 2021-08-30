import os
import argparse
import scipy.stats as st

from .. import params as P
from . import utils

# This is a utility script that takes a path to a desired result folder, reads the results saved in this
# folder, and then computes convergence epochs for a given metric used in the training process.
def run_dispconv(path, crit, lb, seeds, ci_levels, convthresholds):
	for ct in convthresholds:
		print(str(100*ct) + "% CONVERGENCE THRESHOLD" )
		print("ITER, RESULT")
		values = []
		for id in seeds:
			checkpoint_folder = os.path.join(path, 'iter' + str(id), 'checkpoints')
			checkpoint_list = utils.get_checkpoint_list(checkpoint_folder)
			checkpoint_id = max(checkpoint_list)
			checkpoint_file_path = os.path.join(checkpoint_folder, "checkpoint" + str(checkpoint_id) + ".pt")
			loaded_checkpoint = utils.load_dict(checkpoint_file_path)
			series = loaded_checkpoint['val_result_data'][loaded_checkpoint['crit_names'].index(crit)]
			best_res = max(series)
			conv_epoch = 0
			hb = not lb
			for i in range(len(series)):
				if utils.is_converged(series[i], best_res, ct, hb):
					conv_epoch = i
					break
			values.append(conv_epoch)
			print(str(id) + ", " + str(conv_epoch))
		avg = sum(values)/len(values)
		se = st.sem(values)
		print("AVG, " + str(avg))
		for ci_lvl in ci_levels:
			ci = st.t.interval(ci_lvl, len(values) - 1, loc=avg, scale=se)
			ci_str = "+/- " + str((ci[1] - ci[0])/2)
			print(str(ci_lvl*100) + "% CI, " + ci_str)
			

if __name__ == '__main__':
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--path', default=P.DEFAULT_DISPCONV_PATH, help="Path to the csv file you want to examine.")
	parser.add_argument('--crit', default=P.DEFAULT_CRIT, help="Name of criterion you want to evaluate.")
	parser.add_argument('--lb', action='store_true', default=P.DEFAULT_DISPCONV_LB, help="Pass this flag if you want to evaluate a lower-is-better criterion (False by default).")
	parser.add_argument('--seeds', nargs='*', default=P.DEFAULT_SEEDS, type=int, help="The RNG seeds of the experiments you want to examine.")
	parser.add_argument('--ci', nargs='*', default=P.DEFAULT_CI_LEVELS, type=float, help="Confidence interval levels you want to report.")
	parser.add_argument('--convthresholds', nargs='*', default=P.DEFAULT_CONVTHRESHOLDS, type=float, help="Convergence threshold levels you want to report.")
	args = parser.parse_args()
	
	run_dispconv(path=args.path, crit=args.crit, lb=args.lb, seeds=args.seeds, ci_levels=args.ci, convthresholds=args.convthresholds)