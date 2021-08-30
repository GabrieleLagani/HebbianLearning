import argparse
import csv
import scipy.stats as st

from .. import params as P

# This is a utility script that takes a path to a desired result folder, reads the results saved in this
# folder, and then computes mean, confidence intervals and general stats.
def run_dispstats(path, seeds, ci_levels):
	with open(path, 'r') as csv_file:
		reader = csv.reader(csv_file)
		d = dict(reader)
		values = []
		print("ITER, RESULT")
		for id in seeds:
			v = d[str(id)]
			print(str(id) + ", " + v)
			values.append(float(v))
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
	parser.add_argument('--path', default=P.DEFAULT_DISPSTATS_PATH, help="Path to the csv file you want to examine.")
	parser.add_argument('--seeds', nargs='*', default=P.DEFAULT_SEEDS, type=int, help="The RNG seeds of the experiments you want to examine.")
	parser.add_argument('--ci', nargs='*', default=P.DEFAULT_CI_LEVELS, type=float, help="Confidence interval levels you want to report.")
	args = parser.parse_args()
	
	run_dispstats(path=args.path, seeds=args.seeds, ci_levels=args.ci)