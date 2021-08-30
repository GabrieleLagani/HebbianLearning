import argparse
import os
import shutil
import torch

from . import params as P
from . import utils
from .experiment import launch_experiment
from .hpsearch import HPSearch


def run_experiment(config, mode, seeds, tokens, hpsearch, hpseeds, checkpoint, restart, branch):
	mode = [P.MODE_TRN, P.MODE_TST] if mode == P.MODE_TRNTST else [mode]
	
	for m in mode:
		print("\n#### CONFIGURATION: " + config + " (mode: " + m + ") ####\n")
		
		# If this is not a hyperparam search config, or this is just a test experiment, run single config
		if not hpsearch or m == P.MODE_TST:
			for i in range(len(seeds)):
				print("\n**** ITERATION ID: " + str(seeds[i]) + " ****\n")
				# Prepare configuration
				CONFIG_OPTIONS = utils.retrieve(config)
				curr_config = utils.Config(config_id=config, config_options=CONFIG_OPTIONS, mode=m,
				                           iter_num=i, iter_id=seeds[i], result_base_folder=os.path.join(P.RESULT_FOLDER, config.replace('.', os.sep)),
				                           tokens=tokens[i % len(tokens)] if tokens is not None else None,
				                           summary="Iteration id: " + str(seeds[i]))
				# Branch from other configuration, if required
				if branch is not None and branch != curr_config.CONFIG_ID:
					BRANCH_CONFIG_OPTIONS = utils.retrieve(branch)
					branch_config = utils.Config(config_id=branch, config_options=BRANCH_CONFIG_OPTIONS, mode=m,
					                             iter_num=i, iter_id=seeds[i], result_base_folder=os.path.join(P.RESULT_FOLDER, branch.replace('.', os.sep)),
					                             tokens=tokens[i % len(tokens)] if tokens is not None else None,
					                             summary="Iteration id: " + str(seeds[i]))
					if os.path.exists(branch_config.RESULT_FOLDER) and os.path.isdir(branch_config.RESULT_FOLDER):
						if os.path.exists(curr_config.RESULT_FOLDER) and os.path.isdir(curr_config.RESULT_FOLDER):
							shutil.rmtree(curr_config.RESULT_FOLDER)
						shutil.copytree(branch_config.RESULT_FOLDER, curr_config.RESULT_FOLDER)
				# Run experiment instance
				launch_experiment(config=curr_config, checkpoint=checkpoint, restart=restart)
		
		# Otherwise, run hyperparam search
		else:
			HPEXP_RESULT_BASE_FOLDER = os.path.join(P.HPEXP_RESULT_FOLDER, config.replace('.', os.sep))
			HPEXP_SUMMARY_LOG_PATH = os.path.join(HPEXP_RESULT_BASE_FOLDER, 'hpexp_log.txt')
			HPEXP_RESULT_CSV_PATH = os.path.join(HPEXP_RESULT_BASE_FOLDER, 'hpexp_results.csv')
			best_result = None
			best_hyperparams = None
			best_hpseed = None
			if restart:
				if os.path.exists(HPEXP_RESULT_BASE_FOLDER) and os.path.isdir(HPEXP_RESULT_BASE_FOLDER): shutil.rmtree(HPEXP_RESULT_BASE_FOLDER)
			logger = utils.Logger(HPEXP_SUMMARY_LOG_PATH)
			logger.clear()
			
			for i in range(len(hpseeds)):
				logger.print_and_log("")
				logger.print_and_log("==== HYPERPARAMETER SEARCH WITH HP SEED " + str(hpseeds[i]) + " ====")
				logger.print_and_log("Best hyperparameter result so far: {}".format(best_result) + " with hp seed: " + str(best_hpseed) + " with hyperparameters:\n" + str(best_hyperparams) + "\n")
				
				# Initialize hyperparam search
				hps = HPSearch(config=config, hpsearch_num=i, hpseed=hpseeds[i], seeds=seeds, tokens=tokens,
				               hpexp_summary="Best hyperparameter result so far: {}".format(best_result) + " with hp seed: " + str(best_hpseed) + " with hyperparameters:\n" + str(best_hyperparams))
				# If not --restart, try to load checkpoint, if available, and recover hpsearch state from this checkpoint
				if not restart:
					loaded_checkpoint = hps.recover_checkpoint()
					if loaded_checkpoint is not None: hps = utils.dict2obj(loaded_checkpoint, hps)
				
				# Run hyperparam search
				hps.run_hpsearch()
				
				# Retrieve results
				result = hps.get_best_result()
				hyperparams = hps.get_best_hyperparams()
				logger.print_and_log("Best hyperparameter result so far: {}".format(best_result) + " with hp seed: " + str(best_hpseed) + " with hyperparameters:\n" + str(best_hyperparams))
				logger.print_and_log("Current hyperparameter result: {}".format(result) + " with hp seed: " + str(hpseeds[i]) + " with hyperparameters:\n" + str(hyperparams))
				
				if utils.is_better(result, best_result, P.HIGHER_IS_BETTER):
					logger.print_and_log("New best hyperparameters found!")
					best_result = result
					best_hyperparams = hyperparams
					best_hpseed = hpseeds[i]
				
				# Save results to csv
				utils.update_csv(hpseeds[i], result, HPEXP_RESULT_CSV_PATH)
			
			# Copy best hyperparameter results to config result folder
			HPEXP_BEST_RESULT_FOLDER = os.path.join(HPEXP_RESULT_BASE_FOLDER, 'hpseed' + str(best_hpseed) + '_best')
			CONFIG_RESULT_FOLDER = os.path.join(P.RESULT_FOLDER, config.replace('.', os.sep))
			logger.print_and_log("")
			logger.print_and_log("Copying best hyperparameter result folder to configuration result folder...")
			if os.path.exists(CONFIG_RESULT_FOLDER) and os.path.isdir(CONFIG_RESULT_FOLDER):
				shutil.rmtree(CONFIG_RESULT_FOLDER)
			shutil.copytree(HPEXP_BEST_RESULT_FOLDER, CONFIG_RESULT_FOLDER)
			logger.print_and_log("Finished!\n")
	

def main():
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', default=P.DEFAULT_CONFIG, help="The experiment configuration you want to run.")
	parser.add_argument('--mode', default=P.DEFAULT_MODE, choices=[P.MODE_TRN, P.MODE_TST, P.MODE_TRNTST], help="Whether you want to run a train or test experiment.")
	parser.add_argument('--device', default=P.DEVICE, help="The device you want to use for the experiment.")
	parser.add_argument('--seeds', nargs='*', default=P.DEFAULT_SEEDS, type=int, help="The RNG seeds you want to use for the experiment.")
	parser.add_argument('--tokens', nargs='*', default=P.DEFAULT_TOKENS, help="A list of strings to be replaced in special configuration options.")
	parser.add_argument('--hpsearch', action='store_true', default=P.DEFAULT_HPSEARCH, help="Whether you want to run an hyperparameter search experiment on the selected configuration.")
	parser.add_argument('--hpseeds', nargs='*', default=P.DEFAULT_HPSEEDS, type=int, help="The RNG seeds you want to use for hyperparameter search.")
	parser.add_argument('--dataseeds', nargs='*', default=P.GLB_PARAMS[P.KEY_GLB_DATASEEDS], type=int, help="The RNG seeds you want to use for dataset random splitting and statistics computation.")
	parser.add_argument('--checkpoint', default=P.DEFAULT_CHECKPOINT, help="A checkpoint id from which to resume training.")
	parser.add_argument('--restart', action='store_true', default=P.DEFAULT_RESTART, help="Whether you want to restart the experiment from scratch, overwriting previous checkpoints in the save path.")
	parser.add_argument('--clearhist', action='store_true', default=P.CLEARHIST, help="Whether you want to keep the last saved checkpoint only and clear the old checkpoint history.")
	parser.add_argument('--branch', default=P.DEFAULT_BRANCH, help="A previous experiment configuration you want to branch from.")
	args = parser.parse_args()
	
	# Get device
	device = args.device
	# Check that selected device is available, otherwise switch to cpu
	if device != 'cpu':
		try: torch.cuda.get_device_name(device)
		except:
			print("Warning: selected device " + device + " not available. Switching to cpu.")
			device = 'cpu'
	
	# Override default params
	P.DEVICE = device
	P.CLEARHIST = args.clearhist
	P.GLB_PARAMS[P.KEY_GLB_DATASEEDS] = args.dataseeds
	
	run_experiment(args.config, args.mode, args.seeds, args.tokens, args.hpsearch, args.hpseeds, args.checkpoint, args.restart, args.branch)

if __name__ == '__main__':
	main()
	
