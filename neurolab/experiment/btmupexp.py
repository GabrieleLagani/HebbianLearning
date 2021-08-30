import os

from .experiment import Experiment, launch_experiment
from .. import params as P
from .. import utils


# Experiment class for Bottom-Up experiments, in which a list of sub-experiments is performed in sequence.
# Each experiment trains part of a composed model in a bottom-up fashion.
class BtmUpExperiment(Experiment):
	def __init__(self, config):
		self.config = config
		self.logger = utils.Logger(self.config.LOG_PATH)
		
		# State variables
		self.restart = True
		self.best_result = None
	
	def state_dict(self):
		return {
			'restart': self.restart,
			'best_result': self.best_result,
		}
	
	def load_state_dict(self, d):
		self.restart = d['restart']
		self.best_result = d['best_result']
	
	# Prepare experiment by loading various components needed during execution
	def prepare(self):
		pass
	
	# Load model objects to be used in the experiment
	def load_models(self):
		pass
	
	# Load all the components related to model optimization
	def load_optimizer(self):
		pass
	
	# Save plots
	def save_plots(self):
		pass
	
	# Save files resulting from a model evaluation
	def save_results(self):
		pass
	
	# Restore possibly corrupted saved files after a previous crash
	def recover_saved_files(self):
		pass
	
	# Mothod containing logic of an evaluation pass
	def eval_pass(self):
		pass
	
	# Method containing logic of a training pass
	def train_pass(self):
		pass
	
	# Method containing schedule updating logic
	def schedule(self):
		pass
	
	def get_config(self, idx, config):
		CONFIG_OPTIONS = utils.retrieve(config)
		# Pass specific parameters to sub-configuration
		for key in self.config.CONFIG_OPTIONS.keys():
			if '::' not in key:
				if key != P.KEY_SUBCONFIG_LIST and key != P.KEY_HPMANAGER and key != P.KEY_HPSEARCH_PARAMS and key != P.KEY_EXPERIMENT:
					# Keys without '::' are considered common to all sub-experiments, except for hpsearch and subconfig
					# keys that are for the btmup experiment.. also, note that the experiment type should not be
					# overridden when going from the btmup experiment to the sub-experiments
					CONFIG_OPTIONS[key] = self.config.CONFIG_OPTIONS[key]
			else: # Key in the form e.g. 'exp0+exp1::key_name', representing a key for experiments 0 and 1
				prefix, k = key.split('::', 1) # Split prefix from suffix at the semicolon
				if 'exp' + str(idx) in prefix.split('+'): # Split at '+' and check that the current experiment is among those that need this key
					CONFIG_OPTIONS[k] = self.config.CONFIG_OPTIONS[key]
					if k == P.KEY_PRE_NET_MDL_PATHS or k == P.KEY_NET_MDL_PATHS:
						# If we are passing a model path, in this case, this should be relative to the result base folder.
						# The appropriate result base folder path is automatically prepended.
						if type(CONFIG_OPTIONS[k]) is str: CONFIG_OPTIONS[k] = [CONFIG_OPTIONS[k]]
						CONFIG_OPTIONS[k] = [os.path.join(self.config.RESULT_BASE_FOLDER, CONFIG_OPTIONS[k][i]) for i in range(len(CONFIG_OPTIONS[k]))]
		# Generate Config object
		tokens = None if idx == 0 else ','.join(map(str, [((0 if i == 0 else 2) + i) * 100 + self.config.ITER_ID for i in range(idx)])) # Generate tokens for sub-experiment as 0, 300, 400... + base iter_id
		iter_id = ((0 if idx == 0 else 2) + idx) * 100 + self.config.ITER_ID # set sub-experiment iter_id to 0, 300, 400, 500... + base iter_id
		return utils.Config(config_id=config, config_options=CONFIG_OPTIONS, mode=self.config.MODE,
                            iter_num=self.config.ITER_NUM, iter_id=iter_id, result_base_folder=os.path.join(self.config.RESULT_BASE_FOLDER, 'exp' + str(idx)),
                            tokens=tokens, summary=self.config.SUMMARY)
		
	# Perform model evaluation
	def run_eval(self):
		# Retrieve configuration and launch experiment
		for idx, config in enumerate(self.config.CONFIG_OPTIONS[P.KEY_SUBCONFIG_LIST]):
			# Execute experiment
			launch_experiment(config=self.get_config(idx, config), checkpoint=None, restart=False)

	# Perform model training
	def run_train(self):
		restart = self.restart # This is going to be True if no checkpoint is available or if --restart, otherwise this is going to be False
		# Save checkpoint where self.restart is set to False, so that if the user reruns the simulation without the
		# --restart flag, restart is going to be False. If instead the --restart Flag is used, the checkpoint is not
		# loaded and restart is going to be True as by default.
		self.restart = False
		utils.save_dict(utils.obj2dict(self), os.path.join(self.config.CHECKPOINT_FOLDER, "checkpoint0.pt"))
		
		# Retrieve configuration and launch experiment
		for idx, config in enumerate(self.config.CONFIG_OPTIONS[P.KEY_SUBCONFIG_LIST]):
			# Execute experiment
			self.best_result = launch_experiment(config=self.get_config(idx, config), checkpoint=None, restart=restart)
	
	# Print epoch progress information
	def print_train_progress(self, start_epoch, current_epoch, total_epochs, elapsed_time):
		pass
	
	# Return best validation result so far
	def get_best_result(self):
		return self.best_result

