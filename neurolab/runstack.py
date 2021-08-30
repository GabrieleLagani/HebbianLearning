import argparse
import torch

from . import params as P
from . import utils
from .runexp import run_experiment


def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--stack', default=P.DEFAULT_STACK, help="The experiment stack you want to run.")
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The default experiment configuration you want to use for your stack.")
	parser.add_argument('--mode', default=P.DEFAULT_MODE, choices=[P.MODE_TRN, P.MODE_TST, P.MODE_TRNTST], help="The default mode you want to use for your stack.")
	parser.add_argument('--device', default=P.DEVICE, help="The default device you want to use for your stack.")
	parser.add_argument('--seeds', nargs='*', default=P.DEFAULT_SEEDS, type=int, help="The default RNG seeds you want to use for your stack.")
	parser.add_argument('--tokens', nargs='*', default=P.DEFAULT_TOKENS, help="A list of strings to be replaced in special configuration options.")
	parser.add_argument('--hpsearch', action='store_true', default=P.DEFAULT_HPSEARCH, help="The default hpsearch flag you want to use for your stack.")
	parser.add_argument('--hpseeds', nargs='*', default=P.DEFAULT_HPSEEDS, type=int, help="The default RNG seeds for hyperparameter search that you want to use for your stack.")
	parser.add_argument('--dataseeds', nargs='*', default=P.GLB_PARAMS[P.KEY_GLB_DATASEEDS], type=int, help="The RNG seeds you want to use for dataset random splitting and statistics computation.")
	parser.add_argument('--checkpoint', default=P.DEFAULT_CHECKPOINT, help="The default checkpoint you want to use for your stack.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="The default restart flag you want to use for your stack.")
	parser.add_argument('--clearhist', action='store_true', default=P.CLEARHIST, help="The default clearhist flag you want to use for your stack.")
	parser.add_argument('--branch', default=P.DEFAULT_BRANCH, help="The default branch you want to use for your stack.")
	args = parser.parse_args()
	
	# retrieve stack and launch every experiment in the stack
	stack = utils.retrieve(args.stack)
	for exp_details in stack:
		# Get device
		device = exp_details.get(P.KEY_STACK_DEVICE, args.device)
		# Check that selected device is available, otherwise switch to cpu
		if device != 'cpu':
			try: torch.cuda.get_device_name(device)
			except:
				print("Warning: selected device " + device + " not available. Switching to cpu.")
				device = 'cpu'
		
		# Override default params
		P.DEVICE = device
		P.CLEARHIST = exp_details.get(P.KEY_STACK_CLEARHIST, args.clearhist)
		P.GLB_PARAMS[P.KEY_GLB_DATASEEDS] = args.dataseeds
		
		run_experiment(exp_details.get(P.KEY_STACK_CONFIG, args.config), exp_details.get(P.KEY_STACK_MODE, args.mode), exp_details.get(P.KEY_STACK_SEEDS, args.seeds), exp_details.get(P.KEY_STACK_TOKENS, args.tokens), exp_details.get(P.KEY_STACK_HPSEARCH, args.hpsearch), exp_details.get(P.KEY_STACK_HPSEEDS, args.hpseeds), exp_details.get(P.KEY_STACK_CHECKPOINT, args.checkpoint), exp_details.get(P.KEY_STACK_RESTART, args.restart), exp_details.get(P.KEY_STACK_BRANCH, args.branch))

if __name__ == '__main__':
	main()
	
