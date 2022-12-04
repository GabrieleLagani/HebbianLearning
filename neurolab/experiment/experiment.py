import time
import os
import copy
import torch

from .. import params as P
from .. import utils


# Base class containing the experiment logic
class Experiment:
	def __init__(self, config):
		self.config = config
		self.logger = utils.Logger(self.config.LOG_PATH)
		
		# For reproducibility
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		# Set RNG seed
		utils.set_rng_seed(self.config.ITER_ID)
		self.rng_state = utils.get_rng_state()
		
		# Initialize experiment state variables
		self.current_epoch = 0 # The current epoch of training process
		self.early_stop = self.config.CONFIG_OPTIONS.get(P.KEY_EARLY_STOP, True)
		self.train_result_data = {} # Dictionary of sequences of training results over time
		self.val_result_data = {} # Dictionary of sequences of validation results over time
		self.last_eval_epoch = 0 # Epoch at which last evaluation occurred
		self.best_result = None # Best evaluation result so far
		self.best_epoch = 0 # Epoch at which best evaluation result occurred
		self.best_model_dicts = {} # Collection of model dictionaries corresponding to the net_list that produced best result
		self.convergence_epochs = {perc: 0 for perc in P.DEFAULT_CONVTHRESHOLDS}
		
		# Determines how often we want to validate. Eval interval schedule is in config options because we might want
		# to save more often for tasks that take longer, hence each task has its own depending on how long it takes.
		# Eval interval schedule is a dictionary {update_step_1: eval_interval_1, update_step_2: eval_interval_2, ...}
		# representing that from update_step_i we want to schedule evaluations every eval_interval_i
		self.eval_interval_schedule = self.config.CONFIG_OPTIONS.get(P.KEY_EVAL_INTERVAL_SCHEDULE, {})
		if 1 not in self.eval_interval_schedule: self.eval_interval_schedule[1] = 1
		self.eval_interval = self.eval_interval_schedule[1]
	
	def state_dict(self):
		self.rng_state = utils.get_rng_state()
		return {
			'rng_state': self.rng_state,
			'current_epoch': self.current_epoch,
			'train_result_data': self.train_result_data,
			'val_result_data': self.val_result_data,
			'last_eval_epoch': self.last_eval_epoch,
			'best_result': self.best_result,
			'best_epoch': self.best_epoch,
			'best_model_dicts': self.best_model_dicts,
			'convergence_epochs': self.convergence_epochs,
			'eval_interval_schedule': self.eval_interval_schedule,
			'eval_interval': self.eval_interval,
			'net_list': [self.net_list[i].state_dict() for i in range(len(self.net_list))],
			'criteria': [utils.obj2dict(c) for c in self.criteria],
			'crit_names': self.crit_names,
			'loss': utils.obj2dict(self.loss),
			'optimizer': utils.obj2dict(self.optimizer),
			'scheduler': utils.obj2dict(self.scheduler),
		}
	
	def load_state_dict(self, d):
		self.rng_state = d['rng_state']
		self.current_epoch = d['current_epoch']
		self.train_result_data = d['train_result_data']
		self.val_result_data = d['val_result_data']
		self.last_eval_epoch = d['last_eval_epoch']
		self.best_result = d['best_result']
		self.best_epoch = d['best_epoch']
		self.best_model_dicts = d['best_model_dicts']
		self.convergence_epochs = d['convergence_epochs']
		self.eval_interval_schedule = d['eval_interval_schedule']
		self.eval_interval = d['eval_interval']
		self.logger.print_and_log("Recovering network state...")
		for i in range(len(self.net_list)): self.logger.print_and_log(self.net_list[i].load_state_dict(d['net_list'][i], strict=False))
		for i in range(len(self.net_list)): self.net_list[i].to(P.DEVICE)
		for i in range(len(self.pre_net_list)): self.pre_net_list[i].to(P.DEVICE)
		self.logger.print_and_log("Network state recovered!")
		self.logger.print_and_log("Recovering optimization state...")
		utils.set_rng_state(self.rng_state)
		for i in range(self.num_criteria): self.criteria[i] = utils.dict2obj(d['criteria'][i], self.criteria[i])
		self.crit_names = d['crit_names']
		if self.config.MODE == P.MODE_TRN:
			self.loss = utils.dict2obj(d['loss'], self.loss)
			param_groups = []
			for i in range(len(self.net_list)): param_groups += self.net_list[i].get_param_groups()
			self.optimizer = self.optim_manager.get_optimizer(param_groups) if self.optim_manager is not None else None
			self.optimizer = utils.dict2obj(d['optimizer'], self.optimizer)
			self.scheduler = self.sched_manager.get_scheduler(self.optimizer, saved_state=d['scheduler']) if self.sched_manager is not None else None
		self.logger.print_and_log("Optimization state recovered!")
	
	# Transform output of a processing module to input in the form expected by next processing stages
	def select_output(self, outputs, i):
		L = 0
		if self.config.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MODULES, None) is not None: L = len(self.config.CONFIG_OPTIONS[P.KEY_PRE_NET_MODULES])
		which_output = '*' # * gives the whole output dictionary
		if self.config.CONFIG_OPTIONS.get(P.KEY_PRE_NET_OUTPUTS if i < L else P.KEY_NET_OUTPUTS, None) is not None and (i if i < L else (i - L)) < len(self.config.CONFIG_OPTIONS[P.KEY_PRE_NET_OUTPUTS if i < L else P.KEY_NET_OUTPUTS]):
			which_output = self.config.CONFIG_OPTIONS[P.KEY_PRE_NET_OUTPUTS if i < L else P.KEY_NET_OUTPUTS][i if i < L else (i - L)]
		
		# Fetch specific entry of output dictionary, if needed
		if which_output == '*': res = outputs
		elif '+' in which_output: res = {key: outputs[key] for key in which_output.split('+') if key != ''}
		else: res = outputs[which_output]
		
		return res
	
	# Prepare experiment by loading various components needed during execution
	def prepare(self):
		# Prepare network model to be trained
		self.load_models()
		# Prepare optimizer
		self.load_optimizer()
	
	def specify_config(self, config, idx):
		spec_config = copy.deepcopy(config)
		# Pass specific parameters to sub-configuration
		for key in config.CONFIG_OPTIONS.keys():
			if ':' in key: # Key in the form e.g. 'mdl0+mdl1:key_name', representing a key for models 0 and 1
				prefix, k = key.split(':', 1) # Split prefix from suffix at the semicolon
				if 'mdl' + str(idx) in prefix.split('+'): # Split at '+' and check that the current experiment is among those that need this key
					spec_config.CONFIG_OPTIONS[k] = copy.deepcopy(config.CONFIG_OPTIONS[key]) # spec_config.CONFIG_OPTIONS[key]
		return spec_config
	
	# Load model objects to be used in the experiment
	def load_models(self):
		self.logger.print_and_log("Preparing network...")
		
		# Load pre-processing network
		pre_net_list = []
		pre_net_modules = self.config.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MODULES, None)
		pre_net_mdl_paths = self.config.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MDL_PATHS, None)
		if pre_net_modules is not None:
			for i in range(len(pre_net_modules)):
				# Load pre-processing network if needed
				input_shape = self.config.CONFIG_OPTIONS.get(P.KEY_INPUT_SHAPE, None)
				if i > 0: input_shape = utils.tens2shape(self.select_output(pre_net_list[i - 1].get_dummy_fmap(fwd=True), i - 1))
				pre_net_list += [utils.retrieve(pre_net_modules[i])(config=self.specify_config(self.config, i), input_shape=input_shape)]
				if pre_net_mdl_paths is not None and i < len(pre_net_mdl_paths) and pre_net_mdl_paths[i] is not None:
					self.logger.print_and_log("Searching for available saved model for pre-network " + str(i) + "...")
					pre_net_state = utils.load_dict(os.path.normpath(pre_net_mdl_paths[i]))
					if pre_net_state is not None:
						pre_net_list[i].load_state_dict(pre_net_state)
						self.logger.print_and_log("Pre-network model loaded!")
					else:
						self.logger.print_and_log("No valid saved model found for pre-network")
						raise FileNotFoundError("No valid saved model found for pre-network")
				for p in pre_net_list[i].parameters(): p.requires_grad = False
				pre_net_list[i].eval()
		self.pre_net_list = pre_net_list
		
		# Load network
		input_shape = self.config.CONFIG_OPTIONS.get(P.KEY_INPUT_SHAPE, None)
		if len(pre_net_list) > 0: input_shape = utils.tens2shape(self.select_output(pre_net_list[len(pre_net_list) - 1].get_dummy_fmap(fwd=True), len(pre_net_list) - 1))
		net_list = []
		net_modules = self.config.CONFIG_OPTIONS[P.KEY_NET_MODULES]
		net_mdl_paths = self.config.CONFIG_OPTIONS.get(P.KEY_NET_MDL_PATHS, None)
		testing_saved_model = (self.config.MODE == P.MODE_TST and self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS] > 0)
		for i in range(len(net_modules)):
			# Load network models
			if i > 0: input_shape = utils.tens2shape(self.select_output(net_list[i - 1].get_dummy_fmap(fwd=True), len(pre_net_list) + i - 1))
			net_list += [utils.retrieve(net_modules[i])(config=self.specify_config(self.config, len(pre_net_list) + i), input_shape=input_shape)]
			if testing_saved_model or (net_mdl_paths is not None and i < len(net_mdl_paths) and net_mdl_paths[i] is not None):
				# Load network model to be tested or pre-trained network model for fine tuning
				self.logger.print_and_log("Searching for available saved model for network " + str(i) + "...")
				loaded_model = utils.load_dict(self.config.SAVED_MDL_PATHS[i] if testing_saved_model else net_mdl_paths[i])
				if loaded_model is not None:
					net_list[i].load_state_dict(loaded_model)
					self.logger.print_and_log("Model loaded!")
				else:
					self.logger.print_and_log("No valid saved model found")
					raise FileNotFoundError("No valid saved model found")
		self.net_list = net_list
		
		# Move all models to required device
		for i in range(len(self.pre_net_list)): self.pre_net_list[i].to(P.DEVICE)
		for i in range(len(self.net_list)): self.net_list[i].to(P.DEVICE)
		
		# Model loading completed
		self.logger.print_and_log("Network ready!")
	
	# Load all the components related to model optimization
	def load_optimizer(self):
		# Prepare evaluation criteria: This is a list of objectives to be evaluated during the experiment.
		# In particular, the first one is used as reference criterion to estabilish the best model.
		crit_manager_class = self.config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER]
		self.num_criteria = len(crit_manager_class)
		self.crit_managers = [utils.retrieve(crit_manager_class[i])(self.config) for i in range(self.num_criteria)]
		P.HIGHER_IS_BETTER = self.crit_managers[0].higher_is_better() # Set the HIGHER_IS_BETTER global flag in params. By default it is True.
		self.criteria = [self.crit_managers[i].get_metric() for i in range(self.num_criteria)]
		self.crit_names = [self.crit_managers[i].get_name() for i in range(self.num_criteria)]
		for i in range(self.num_criteria):
			self.train_result_data[i] = {}
			self.val_result_data[i] = {}
		
		self.loss_manager = None
		self.optim_manager = None
		self.sched_manager = None
		self.loss = None
		self.optimizer = None
		self.scheduler = None
		if self.config.MODE == P.MODE_TRN:
			self.logger.print_and_log("Preparing optimizer...")
			
			# Prepare loss function. This is an objective which is going to be minimized by backpropagation during training.
			loss_manager_class = self.config.CONFIG_OPTIONS.get(P.KEY_LOSS_METRIC_MANAGER, None)
			self.loss_manager = utils.retrieve(loss_manager_class)(self.config) if loss_manager_class is not None else None
			self.loss = self.loss_manager.get_metric() if self.loss_manager is not None else None
			
			# Prepare optimizer and scheduler.
			optim_manager_class = self.config.CONFIG_OPTIONS.get(P.KEY_OPTIM_MANAGER, None)
			self.optim_manager = utils.retrieve(optim_manager_class)(self.config) if optim_manager_class is not None else None
			param_groups = []
			for i in range(len(self.net_list)): param_groups += self.net_list[i].get_param_groups()
			self.optimizer = self.optim_manager.get_optimizer(param_groups) if self.optim_manager is not None else None
			sched_manager_class = self.config.CONFIG_OPTIONS.get(P.KEY_SCHED_MANAGER, None) if self.optim_manager is not None else None
			self.sched_manager = utils.retrieve(sched_manager_class)(self.config) if sched_manager_class is not None else None
			self.scheduler = self.sched_manager.get_scheduler(self.optimizer) if self.sched_manager is not None else None
			
			self.logger.print_and_log("Optimizer ready!")
	
	# Save plots
	def save_plots(self):
		for i in range(self.num_criteria):
			utils.save_trn_curve_plot(self.train_result_data[i], self.val_result_data[i],
			                          os.path.join(self.config.FIGURE_FOLDER, self.crit_managers[i].get_name() + '.png'),
			                          label=self.crit_managers[i].get_name())
	
	# Save files resulting from a model evaluation
	def save_results(self):
		curr_res = self.val_result_data[0][self.current_epoch]
		self.logger.print_and_log("Best result (" + self.crit_managers[0].get_name() + ") so far: {} at epoch {}/{}".format(self.best_result, self.best_epoch, self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]))
		self.logger.print_and_log("Current result (" + self.crit_managers[0].get_name() + "): {}".format(curr_res))
		# If validation result has improved update best result stats and best model
		if utils.is_better(curr_res, self.best_result, P.HIGHER_IS_BETTER):
			self.logger.print_and_log("Best result (" + self.crit_managers[0].get_name() + ") improved!")
			# Update best result info
			self.best_result = curr_res
			self.best_epoch = self.current_epoch
			utils.update_csv(self.config.ITER_ID, self.best_epoch, os.path.join(self.config.RESULT_BASE_FOLDER, 'convergence_epoch.csv'), ci_levels=P.DEFAULT_CI_LEVELS)
			# Update convergence time info
			for perc in self.convergence_epochs.keys():
				for i in range(self.convergence_epochs[perc], self.current_epoch + 1):
					if i in self.val_result_data[0] and utils.is_converged(self.val_result_data[0][i], self.best_result, perc, P.HIGHER_IS_BETTER):
						self.convergence_epochs[perc] = i
						break
				self.logger.print_and_log("{}% convergence epoch {}/{}".format(100 * perc, self.convergence_epochs[perc], self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]))
				utils.update_csv(self.config.ITER_ID, self.convergence_epochs[perc], os.path.join(self.config.RESULT_BASE_FOLDER, 'convergence_epoch_' + str(100*perc) + '_perc.csv'), ci_levels=P.DEFAULT_CI_LEVELS)
			# Convert model to dict and save
			self.logger.print_and_log("Saving new best models...")
			for i in range(len(self.net_list)):
				self.best_model_dicts[i] = copy.deepcopy(self.net_list[i].state_dict())
				if self.early_stop: utils.save_dict(self.best_model_dicts[i], self.config.SAVED_MDL_PATHS[i])
			self.logger.print_and_log("Models saved!")
		else:
			for perc in self.convergence_epochs.keys(): self.logger.print_and_log("{}% convergence epoch {}/{}".format(100 * perc, self.convergence_epochs[perc], self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]))
		if not self.early_stop:
			# Convert model to dict and save
			self.logger.print_and_log("Saving new models...")
			for i in range(len(self.net_list)):
				utils.save_dict(self.net_list[i].state_dict(), self.config.SAVED_MDL_PATHS[i])
			self.logger.print_and_log("Models saved!")
			
		# Save plots
		self.logger.print_and_log("Saving plots...")
		self.save_plots()
		self.logger.print_and_log("Plots saved!")
	
	# Restore possibly corrupted saved files after a previous crash
	def recover_saved_files(self):
		utils.update_csv(self.config.ITER_ID, self.best_epoch, os.path.join(self.config.RESULT_BASE_FOLDER, 'convergence_epoch.csv'), ci_levels=P.DEFAULT_CI_LEVELS)
		for perc in self.convergence_epochs.keys():
			utils.update_csv(self.config.ITER_ID, self.convergence_epochs[perc], os.path.join(self.config.RESULT_BASE_FOLDER, 'convergence_epoch_' + str(100*perc) + '_perc.csv'), ci_levels=P.DEFAULT_CI_LEVELS)
		for i in range(len(self.net_list)):
			if self.early_stop: utils.save_dict(self.best_model_dicts[i], self.config.SAVED_MDL_PATHS[i])
			else: utils.save_dict(self.net_list[i].state_dict(), self.config.SAVED_MDL_PATHS[i])
		self.save_plots()
	
	# Method containing logic of an evaluation pass
	def eval_pass(self):
		raise NotImplementedError
	
	# Method containing logic of a training pass
	def train_pass(self):
		raise NotImplementedError
	
	# Method containing schedule updating logic
	def schedule(self):
		if self.scheduler is not None: self.schedule_optimizer()
		if (self.current_epoch + 1) in self.eval_interval_schedule.keys(): self.eval_interval = self.eval_interval_schedule[self.current_epoch + 1]
	
	# Logic for the rl scheduling policy
	def schedule_optimizer(self):
		if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
			self.scheduler.step(self.val_result_data[0][self.current_epoch])
		else:
			self.scheduler.step()
	
	# Logic for gradient pre-conditioning to apply after gradients are computed but before the optimization step is
	# taken (e.g. gradient norm clipping). By default, it does nothing.
	def precond_grads(self):
		pass
	
	# Perform model evaluation
	def run_eval(self):
		self.logger.print_and_log("\nTesting...")
		test_res = self.eval_pass()
		for i in range(self.num_criteria):
			self.logger.print_and_log("Test result (" + self.crit_managers[i].get_name() + ") of the model: {}".format(test_res[i]))
		
		self.logger.print_and_log("Saving test results...")
		for i in range(self.num_criteria):
			utils.update_csv(self.config.ITER_ID, test_res[i], os.path.join(self.config.RESULT_BASE_FOLDER, self.crit_managers[i].get_name() + '.csv'), ci_levels=P.DEFAULT_CI_LEVELS)
		self.logger.print_and_log("Saved!")
		self.logger.print_and_log("")
		self.logger.print_and_log("--------")
		self.logger.print_and_log("")

	# Perform model training
	def run_train(self):
		# Train the network
		self.logger.print_and_log("")
		self.logger.print_and_log("Starting training..." if self.current_epoch == 0 else "Resuming training from checkpoint " + str(self.current_epoch) + "...")
		start_epoch = self.current_epoch + 1
		start_time = time.time()
		for epoch in range(start_epoch, self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS] + 1):
			self.current_epoch = epoch

			# Print overall progress information at each epoch
			self.print_train_progress(start_epoch=start_epoch, current_epoch=self.current_epoch,
			                          total_epochs=self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS],
			                          elapsed_time=time.time() - start_time)
			
			# Training pass
			self.logger.print_and_log("Training...")
			train_res = self.train_pass()
			for i in range(self.num_criteria):
				self.train_result_data[i][self.current_epoch] = train_res[i]
				self.logger.print_and_log("Training result (" + self.crit_managers[i].get_name() + "): {}".format(train_res[i]))
			
			# Periodically perform validation pass
			if self.current_epoch - self.last_eval_epoch >= self.eval_interval or self.current_epoch == self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]:
				self.logger.print_and_log("Validating...")
				val_res = self.eval_pass()
				for i in range(self.num_criteria):
					self.val_result_data[i][self.current_epoch] = val_res[i]
					self.logger.print_and_log("Validation result (" + self.crit_managers[i].get_name() + "): {}".format(val_res[i]))
				self.save_results()
				self.last_eval_epoch = self.current_epoch
				
			# Update schedulers
			self.schedule()
			
			self.logger.print_and_log("Epoch " + str(self.current_epoch) + "/" + str(self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]) + " completed!")
	
			# Periodically save checkpoint
			if (self.current_epoch % P.CHECKPOINT_INTERVAL == 0) or (self.current_epoch == self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]):
				self.logger.print_and_log("Saving checkpoint " + str(self.current_epoch) + "...")
				utils.save_dict(utils.obj2dict(self), os.path.join(self.config.CHECKPOINT_FOLDER, "checkpoint" + str(self.current_epoch) + ".pt"))
				self.logger.print_and_log("Checkpoint " + str(self.current_epoch) + " saved!")
				# Clear old checkpoint files, if required
				utils.clear_checkpoints(checkpoint_folder=self.config.CHECKPOINT_FOLDER, latest_checkpoint=self.current_epoch, clearhist=P.CLEARHIST)
		
		self.logger.print_and_log("")
		self.logger.print_and_log("Finished!")
		self.logger.print_and_log("Best epoch: {}/{}".format(self.best_epoch, self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]) + ", with results: ")
		for i in range(self.num_criteria): self.logger.print_and_log(self.crit_managers[i].get_name() + ": {}".format(self.val_result_data[i][self.best_epoch]))
		for perc in self.convergence_epochs.keys(): self.logger.print_and_log("{}% convergence epoch: {}/{}".format(100 * perc, self.convergence_epochs[perc], self.config.CONFIG_OPTIONS[P.KEY_NUM_EPOCHS]))
		self.logger.print_and_log("")
		self.logger.print_and_log("--------")
		self.logger.print_and_log("")
	
	# Print epoch progress information
	def print_train_progress(self, start_epoch, current_epoch, total_epochs, elapsed_time):
		self.logger.print_and_log("\nEPOCH " + str(current_epoch) + "/" + str(total_epochs))
		self.logger.print_and_log("Experiment configuration: " + self.config.CONFIG_ID)
		if self.config.SUMMARY is not None: self.logger.print_and_log(self.config.SUMMARY)
	
		elapsed_time_str = "-"
		avg_epoch_duration_str = "-"
		exp_remaining_time_str = "-"
		elapsed_epochs = current_epoch - start_epoch
		if elapsed_epochs > 0:
			avg_epoch_duration = elapsed_time / elapsed_epochs
			remaining_epochs = total_epochs - (current_epoch - 1)
			elapsed_time_str = utils.format_time(elapsed_time)
			avg_epoch_duration_str = utils.format_time(avg_epoch_duration)
			exp_remaining_time_str = utils.format_time(remaining_epochs * avg_epoch_duration)
		self.logger.print_and_log("Elapsed time: " + elapsed_time_str)
		self.logger.print_and_log("Average epoch duration: " + avg_epoch_duration_str)
		self.logger.print_and_log("Expected remaining time: " + exp_remaining_time_str)
		
		last_train_res = ""
		last_val_res = ""
		best_res = ""
		if current_epoch > 1:
			for i in range(self.num_criteria):
				last_train_res += self.crit_managers[i].get_name() + " " + str(self.train_result_data[i][current_epoch - 1]) + ", "
				last_val_res += self.crit_managers[i].get_name() + " " + str(self.val_result_data[i][self.last_eval_epoch]) + ", "
				best_res += self.crit_managers[i].get_name() + " " + str(self.val_result_data[i][self.best_epoch]) + ", "
			for perc in self.convergence_epochs.keys(): best_res += "{}% convergence epoch: {}/{}, ".format(100 * perc, self.convergence_epochs[perc], total_epochs)
		else:
			last_train_res = "-"
			last_val_res = "-"
			best_res = "-"
		self.logger.print_and_log("Last training results: " + last_train_res)
		self.logger.print_and_log("Last validation epoch: {}/{}".format(self.last_eval_epoch, total_epochs) + ", with results: " + last_val_res)
		self.logger.print_and_log("Best epoch so far: {}/{}".format(self.best_epoch, total_epochs) + ", with results: " + best_res)
		
	# Return best validation result so far
	def get_best_result(self):
		return self.best_result


# Method for launching an experiment configuration
def launch_experiment(config, checkpoint, restart):
	# Search for checkpoint file
	checkpoint_id = None
	loaded_checkpoint = None
	if config.MODE == P.MODE_TRN:
		os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
		checkpoint_list = utils.get_checkpoint_list(config.CHECKPOINT_FOLDER)
		if restart and checkpoint is None:
			# Upon restart, clear checkpoint folder
			for c in checkpoint_list: os.remove(os.path.join(config.CHECKPOINT_FOLDER, "checkpoint" + str(c) + ".pt"))
		else:
			if checkpoint is None:
				# Resume from latest checkpoint, by default, if available. If no checkpoint available, set checkpoint file to None and start from scratch
				checkpoint_id = max(checkpoint_list) if len(checkpoint_list) > 0 else None
			else:
				# Resume from a specific checkpoint, if provided.
				checkpoint_id = checkpoint
			# We have determined the checkpoint_file we want to restart from. If it is None, then we will just restart from scratch.
			# If it is not None, then let's try to load the checkpoint from the checkpoint_file.
			if checkpoint_id is not None:
				checkpoint_file_path = os.path.join(config.CHECKPOINT_FOLDER, "checkpoint" + str(checkpoint_id) + ".pt")
				loaded_checkpoint = utils.load_dict(checkpoint_file_path)
				if loaded_checkpoint is not None:
					# Clear old checkpoint files, if required
					utils.clear_checkpoints(checkpoint_folder=config.CHECKPOINT_FOLDER, latest_checkpoint=checkpoint_id, clearhist=P.CLEARHIST)
				else:
					# Checkpoint file not found. Let's warn the user and quit.
					print("Checkpoint file " + checkpoint_file_path + " not found or invalid, please provide a valid checkpoint or start from scratch.")
					exit()
	
	# Initialize experiment
	torch.set_grad_enabled(False) # Gradient computation is disabled by default and enabled only when needed
	RetrievedExperimentClass = utils.retrieve(config.CONFIG_OPTIONS[P.KEY_EXPERIMENT])
	logger = utils.Logger(config.LOG_PATH)
	if config.MODE == P.MODE_TST or (config.MODE == P.MODE_TRN and loaded_checkpoint is None): logger.clear()
	if loaded_checkpoint is not None:
		logger.print_and_log("----")
		logger.print_and_log("Recovering experiment state from checkpoint " + str(checkpoint_id) +  "\n")
	logger.print_and_log("Experiment configuration: " + config.CONFIG_ID)
	logger.print_and_log("Initializing experiment...")
	experiment = RetrievedExperimentClass(config)
	experiment.prepare()
	logger.print_and_log("Experiment initialized!")
	
	# Restore state from checkpoint, if available
	if loaded_checkpoint is not None:
		logger.print_and_log("Restoring saved state from checkpoint " + str(checkpoint_id) +  "...")
		experiment = utils.dict2obj(loaded_checkpoint, experiment)
		experiment.recover_saved_files()
		logger.print_and_log("State restored from checkpoint " + str(checkpoint_id) +  "!")
	logger.log("")
	
	# Log experiment details
	logger.log("Configuration details:\n" + config.CONFIG_INFO)
	logger.log("System details:\n" + config.SYS_INFO)
	
	# Launch experiment
	if config.MODE == P.MODE_TRN:
		experiment.run_train()
		return experiment.get_best_result()
	else:
		experiment.run_eval()

