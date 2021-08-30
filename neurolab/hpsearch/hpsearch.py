import os
import shutil

from .. import params as P
from .. import utils
from ..experiment import launch_experiment


# Hyperparameter search manager base class
class HPManager:
	def __init__(self, config_options, hpsearch_num, hpseed):
		self.config_options = config_options
		self.hpsearch_num = hpsearch_num
		self.hpseed = hpseed
		
		prev_rng_state = utils.get_rng_state()
		utils.set_rng_seed(self.hpseed)
		self.rng_state = utils.get_rng_state()
		utils.set_rng_state(prev_rng_state)
	
	def state_dict(self):
		return utils.state_dict(self)
	
	def load_state_dict(self, d):
		utils.load_state_dict(self, d)
	
	def get_next_hyperparams(self):
		raise NotImplementedError
	
	def update(self, result):
		raise NotImplementedError
	
	def __iter__(self):
		return self
	
	def __next__(self):
		# Prepare rng state
		prev_rng_state = utils.get_rng_state()
		utils.set_rng_state(self.rng_state)
		
		# Get next hyperparams
		curr_hyperparams = self.get_next_hyperparams()
		
		# Restore rng state
		self.rng_state = utils.get_rng_state()
		utils.set_rng_state(prev_rng_state)
		
		if curr_hyperparams is None: raise StopIteration
		return curr_hyperparams

# Class containing the logic for coordinating hyperparam search
class HPSearch:
	def __init__(self, config, hpsearch_num, hpseed, seeds, tokens, hpexp_summary=""):
		self.config = config
		self.hpsearch_num = hpsearch_num
		self.hpseed = hpseed
		self.CONFIG_OPTIONS = utils.retrieve(self.config)
		self.HPSEARCH_RESULT_BASE_FOLDER = os.path.join(P.HPEXP_RESULT_FOLDER, self.config.replace('.', os.sep), 'hpseed' + str(self.hpseed))
		self.HPSEARCH_BEST_RESULT_FOLDER = os.path.join(P.HPEXP_RESULT_FOLDER, self.config.replace('.', os.sep), 'hpseed' + str(self.hpseed) + '_best')
		self.HPSEARCH_CHECKPOINT_PATH = os.path.join(self.HPSEARCH_RESULT_BASE_FOLDER, 'hpcheckpoint.pt')
		self.HPSEARCH_LOG_PATH = os.path.join(self.HPSEARCH_RESULT_BASE_FOLDER, 'hpsearch_log.txt')
		self.logger = utils.Logger(self.HPSEARCH_LOG_PATH)
		self.hpexp_summary = hpexp_summary
		
		# State variables
		self.seeds = seeds
		self.tokens = tokens
		self.best_result = None
		self.best_hyperparams = None
		self.hp = utils.retrieve(self.CONFIG_OPTIONS[P.KEY_HPMANAGER])(self.CONFIG_OPTIONS, self.hpsearch_num, self.hpseed)
		self.hp_state_dict = utils.obj2dict(self.hp)
		self.hp_count = 0
		self.clear = False
		self.result_list = []
		self.curr_iter = 0
		
	def state_dict(self):
		d = {}
		d['seeds'] = self.seeds
		d['tokens'] = self.tokens
		d['best_result'] = self.best_result
		d['best_hyperparams'] = self.best_hyperparams
		d['hp_count'] = self.hp_count
		d['clear'] = self.clear
		d['result_list'] = self.result_list
		d['curr_iter'] = self.curr_iter
		d['hp_state_dict'] = self.hp_state_dict
		return d
	
	def load_state_dict(self, d):
		self.seeds = d['seeds']
		self.tokens = d['tokens']
		self.best_result = d['best_result']
		self.best_hyperparams = d['best_hyperparams']
		self.hp_count = d['hp_count']
		self.clear = d['clear']
		self.result_list = d['result_list']
		self.curr_iter = d['curr_iter']
		self.hp_state_dict = d['hp_state_dict']
		self.hp = utils.dict2obj(self.hp_state_dict, self.hp)
	
	def recover_checkpoint(self):
		d = utils.load_dict(self.HPSEARCH_CHECKPOINT_PATH)
		if d is not None:
			if self.seeds != d['seeds']:
				print("The seeds you provided are different from previous seeds. Please use the same seeds or restart a new experiment with --restart.")
				print("Previous seeds: " + str(d['seeds']))
				print("Current seeds: " + str(self.seeds))
				exit()
			self.logger.print_and_log("----")
			self.logger.print_and_log("Recovering state of hyperparameter search with hp seed: " + str(self.hpseed) + " from checkpoint \n")
		return d
		
	def get_best_result(self):
		return self.best_result
	
	def get_best_hyperparams(self):
		return self.best_hyperparams
	
	def run_hpsearch(self):
		# Log experiment details
		HPSEARCH_INFO = ""
		HPSEARCH_INFO += "CONFIG_ID: " + self.config + "\n"
		HPSEARCH_INFO += "CONFIG_OPTIONS: " + str(self.CONFIG_OPTIONS) + "\n"
		HPSEARCH_INFO += "HPSEED_NUM: " + str(self.hpsearch_num) + "\n"
		HPSEARCH_INFO += "HPSEED: " + str(self.hpseed)
		self.logger.log("Hyperparameter search configuration details:\n" + HPSEARCH_INFO)
		self.logger.log("System details:\n" + utils.get_sys_info())
		
		# Iterate over the configurations proposed by the hyperparam search algorithm and execute the corresponding experiment.
		for curr_hyperparams in self.hp:
			
			# Execute experiment for each iteration seed
			for i in range(self.curr_iter, len(self.seeds)):
				self.logger.print_and_log("")
				self.logger.print_and_log("**** HYPERPARAMETER CONFIGURATION " + str(self.hp_count) + " | HP SEED: " + str(self.hpseed) + " | ITERATION ID: " + str(self.seeds[i]) + " ****")
				self.logger.print_and_log("Summary from past  hp search - " + self.hpexp_summary)
				self.logger.print_and_log("Summary of current hp search - Best hyperparameter result so far: {}".format(self.best_result) + " with hyperparameters:\n" + str(self.best_hyperparams))
				self.logger.print_and_log("Current hyperparameters: \n" + str(curr_hyperparams) + "\n")
				
				# Prepare configuration
				for k in curr_hyperparams.keys(): self.CONFIG_OPTIONS[k] = curr_hyperparams[k]
				curr_config = utils.Config(config_id=self.config, config_options=self.CONFIG_OPTIONS, mode=P.MODE_TRN,
				                           iter_num=i, iter_id=self.seeds[i], result_base_folder=self.HPSEARCH_RESULT_BASE_FOLDER,
				                           tokens=self.tokens[i % len(self.tokens)] if self.tokens is not None else None,
				                           summary="Hyperparameter configuration: " + str(self.hp_count) +
											 " | HP seed: " + str(self.hpseed) +
											 " | Iteration id: " + str(self.seeds[i]) +
											 " | Current hyperparameters:\n" + str(curr_hyperparams) +
											 "\nSummary from past  hp search - " + self.hpexp_summary +
											 "\nSummary of current hp search - Best hyperparameter result so far: {}".format(self.best_result) + " with hyperparameters:\n" + str(self.best_hyperparams)
				                           )
				# Check if hp search result folder contains old files from previous hp config iteration and in this case clear folder
				if not self.clear:
					self.logger.print_and_log("Starting execution of iteration " + str(self.seeds[i]) + "...")
					self.logger.print_and_log("Clearing result folder...")
					if os.path.exists(curr_config.RESULT_FOLDER) and os.path.isdir(curr_config.RESULT_FOLDER): shutil.rmtree(curr_config.RESULT_FOLDER)
					self.clear = True
					utils.save_dict(self.state_dict(), self.HPSEARCH_CHECKPOINT_PATH)
					self.logger.print_and_log("Result folder cleared!")
				else: self.logger.print_and_log("Resuming execution of iteration " + str(self.seeds[i]) + "...")
				
				# Run experiment instance and append results
				res = launch_experiment(config=curr_config, checkpoint=None, restart=False)
				self.result_list.append(res)
				self.clear = False
				self.hp_state_dict = utils.obj2dict(self.hp)
				self.curr_iter += 1
				self.logger.print_and_log("Execution of iteration " + str(self.seeds[i]) + " completed!\n")
				
				# Save iteration state
				self.logger.print_and_log("Saving hyperparameter iteration state...")
				utils.save_dict(self.state_dict(), self.HPSEARCH_CHECKPOINT_PATH)
				self.logger.print_and_log("Hyperparameter iteration state saved!")
			
			# hp iteration loop finished, print result
			result = sum(self.result_list)/len(self.result_list)
			self.logger.print_and_log("")
			self.logger.print_and_log("Iteration loop completed for hyperparameter configuration " + str(self.hp_count) + " with hp seed " + str(self.hpseed))
			self.logger.print_and_log("Summary from past  hp search - " + self.hpexp_summary)
			self.logger.print_and_log("Summary of current hp search - Best hyperparameter result so far: {}".format(self.best_result) + " with hyperparameters:\n" + str(self.best_hyperparams))
			self.logger.print_and_log("Summary of current hp search - Current hyperparameter result: {}".format(result) + " with hyperparameters:\n" + str(curr_hyperparams))
			
			# If a new best hp config was found, copy saved results from hp result folder to result folder
			if utils.is_better(result, self.best_result, P.HIGHER_IS_BETTER):
				self.logger.print_and_log("New best hyperparameter configuration found! Saving new best hyperparameter configuration...")
				if os.path.exists(self.HPSEARCH_BEST_RESULT_FOLDER) and os.path.isdir(self.HPSEARCH_BEST_RESULT_FOLDER):
					shutil.rmtree(self.HPSEARCH_BEST_RESULT_FOLDER)
				shutil.copytree(self.HPSEARCH_RESULT_BASE_FOLDER, self.HPSEARCH_BEST_RESULT_FOLDER,
				                ignore=shutil.ignore_patterns(os.path.basename(self.HPSEARCH_LOG_PATH), os.path.basename(self.HPSEARCH_CHECKPOINT_PATH))) # Ignore files to keep track of hp search that are not needed
				self.logger.print_and_log("Hyperparameter configuration saved!")
				self.best_result = result
				self.best_hyperparams = curr_hyperparams
			
			# Update hyperparam search module state
			self.logger.print_and_log("Updating hyperparameter search state...")
			self.hp.update(result)
			self.hp_state_dict = utils.obj2dict(self.hp)
			self.hp_count += 1
			self.result_list = []
			self.curr_iter = 0
			self.logger.print_and_log("Hyperparameter search state updated!")
			
			# Save hp search state. Also save seeds in use, because upon resuming we need to check that the same seeds are used.
			self.logger.print_and_log("Saving hyperparameter search state...")
			utils.save_dict(self.state_dict(), self.HPSEARCH_CHECKPOINT_PATH)
			self.logger.print_and_log("Hyperparameter search state saved!")
		
		self.logger.print_and_log("")
		self.logger.print_and_log("Hyperparameter search with hp seed " + str(self.hpseed) + " finished!")
		self.logger.print_and_log("Summary from past  hp search - " + self.hpexp_summary)
		self.logger.print_and_log("Summary of current hp search - Best hyperparameter result so far: {}".format(self.best_result) + " with hyperparameters:\n" + str(self.best_hyperparams) + "\n")
		
