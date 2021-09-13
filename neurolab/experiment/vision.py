import os
import torch

from .experiment import Experiment
from .. import params as P
from .. import utils

# Object recognition experiment for comparing Hebbian learning with backprop
class VisionExperiment(Experiment):
	def __init__(self, config):
		super().__init__(config)
		
		self.data_manager = None
		self.train_set = None
		self.val_set = None
		self.test_set = None
	
	# Redefine prepare logic to add loading of dataset
	def prepare(self):
		# Prepare dataset
		self.load_dataset()
		# Prepare remaining part of the experiment
		super().prepare()
	
	# Prepare image datasets to be used in the experiment
	def load_dataset(self):
		self.logger.print_and_log("Preparing dataset...")
		
		self.data_manager = utils.retrieve(self.config.CONFIG_OPTIONS[P.KEY_DATA_MANAGER])(self.config)
		self.train_set = self.data_manager.get_train_set()
		self.val_set = self.data_manager.get_val_set()
		self.test_set = self.data_manager.get_test_set()
		
		self.NUM_TRN_SAMPLES = self.data_manager.NUM_TRN_SAMPLES
		self.NUM_VAL_SAMPLES = self.data_manager.NUM_VAL_SAMPLES
		self.NUM_TST_SAMPLES = self.data_manager.NUM_TST_SAMPLES
		
		self.logger.print_and_log("Dataset ready!")
	
	# Redefine evaluation logic to add weight visualization and adversarial evaluation
	def run_eval(self):
		super().run_eval()
		
		# Plot weight visualizations
		KNL_PLT_PATH = os.path.join(self.config.FIGURE_FOLDER, 'kernels.png')
		KNL_INV_PLT_PATH = os.path.join(self.config.FIGURE_FOLDER, 'kernels_inv.png')
		if len(self.pre_net_list) == 0:  # the network is directly attached to the x image
			w = None
			if hasattr(self.net_list[0], 'conv1'): w = self.net_list[0].conv1.weight
			if hasattr(self.net_list[0], 'fc'): w = self.net_list[0].fc.weight.view(-1, *self.net_list.get_input_shape())
			if w is not None:
				self.logger.print_and_log("Plotting weight visualizations...")
				utils.plot_grid(w, KNL_PLT_PATH)
				utils.plot_grid(utils.inv_weight(w), KNL_INV_PLT_PATH)
				self.logger.print_and_log("Plots saved!")
		
		# Add deep layer visualization
		
		# Add adversarial evaluation
	
	# Takes a batch of data as provided by the data manager, and re-organizes it according to what is expected by the type of experiment
	def prepare_batch(self, batch):
		# Get the inputs
		inputs, targets = batch
		
		# Move the inputs to the device in use
		inputs, targets = inputs.to(P.DEVICE), targets.to(P.DEVICE)
		
		# Pre-process inputs
		inputs = self.data_manager.preprocess(inputs)
		
		# Count number of elements in batch
		batch_count = inputs.size(0)
		
		return inputs, targets, batch_count
	
	# Process a batch of data, compute evaluation criteria, optionally a loss metric and weight updates.
	def process_batch(self, batch):
		# Prepare the inputs
		inputs, targets, batch_count = self.prepare_batch(batch)
		
		# Feed inputs to pre-processing networks
		for i in range(len(self.pre_net_list)): inputs = self.select_output(self.pre_net_list[i](inputs), i) # Let the network process the x
		
		# Start gradient tracking if required
		torch.set_grad_enabled(self.loss is not None and self.net_list[-1].training)
		
		outputs = None
		for i in range(len(self.net_list)):
			# Set teacher signal (for local learning rules)
			if self.net_list[i].training: self.net_list[i].set_teacher_signal(targets)
			# Forward step. Select the output from the required layer and prepare it in the correct form expected by successive processing stages
			outputs = self.select_output(self.net_list[i](outputs if i > 0 else inputs), len(self.pre_net_list) + i) # Let the network process the x
			# Unset teacher signal (for local learning rules)
			if self.net_list[i].training: self.net_list[i].set_teacher_signal(None)
		
		# Evaluate loss and perform backpropagation, if required
		if self.optimizer is not None and self.net_list[-1].training:
			self.optimizer.zero_grad()  # Zero out accumulated gradients
			if self.loss is not None:
				loss = self.loss(outputs, targets)  # compute loss
				loss.backward()  # Backward step (compute gradients)
			for i in range(len(self.net_list)): self.net_list[i].local_updates() # Another backward step, apply weight updates computed from local learning rules
			self.optimizer.step()  # Optimize (update weights)
		
		# Stop gradient tracking
		torch.set_grad_enabled(False)
		
		# Compute predicted classes, count number of correct guesses and update variables for keeping track of result.
		batch_res = {i: batch_count * self.criteria[i](outputs, targets) for i in range(self.num_criteria)} # Criteria return averages over batches. We need to keep sums and average only at the end, so we multiply by batch_count
		batch_res = {i: batch_res[i].item() if isinstance(batch_res[i], torch.Tensor) else batch_res[i] for i in range(self.num_criteria)} # Some criteria might return tensors, in this case the tensor is converted to a number.
		
		return batch_res, batch_count

	# Evaluate a model over a dataset
	def eval_pass(self):
		for i in range(len(self.net_list)): self.net_list[i].eval()

		# Variables for computing accuracy
		res = {i: 0 for i in range(self.num_criteria)}  # Collection of results obtained so far
		count = 0  # Number of samples processed so far

		progtracker = utils.ProgressTracker(P.PROGRESS_INTERVAL, self.NUM_VAL_SAMPLES if self.config.MODE == P.MODE_TRN else self.NUM_TST_SAMPLES)
		for batch in (self.val_set if self.config.MODE == P.MODE_TRN else self.test_set):
			# Process batch and compute result and total number of samples in the batch
			batch_res, batch_count= self.process_batch(batch)
			for i in range(self.num_criteria): res[i] += batch_res[i]
			count += batch_count

			# Periodically provide progress information
			progtracker.print_progress(count)

		# Compute final results
		for i in range(self.num_criteria): res[i] = res[i] / count

		return res

	# Perform a training pass over a dataset.
	def train_pass(self):
		# Variables for keeping track of training progress
		count = 0  # total number of samples processed so far
		res = {i: (self.train_result_data[i][self.current_epoch - 1] if len(self.train_result_data[i]) > 0 else 0) for i in range(self.num_criteria)} # training results
		total = int(self.eval_interval * self.NUM_TRN_SAMPLES) # Total number of samples to be processed
		total = self.NUM_TRN_SAMPLES if self.eval_interval >= 1 else (total - (total % self.config.CONFIG_OPTIONS[P.KEY_BATCHSIZE])) # Correct total number of samples based on batch size.
		
		progtracker = utils.ProgressTracker(P.PROGRESS_INTERVAL,  total)
		for batch in self.train_set:
			# Set training mode
			for i in range(len(self.net_list)): self.net_list[i].train()
			
			# Process batch and count number of hits and total number of samples in the batch
			batch_res, batch_count = self.process_batch(batch)
			
			# Unset training mode
			for i in range(len(self.net_list)): self.net_list[i].eval()

			# Update statistics
			count += batch_count
			for i in range(self.num_criteria):
				batch_res_avg = batch_res[i] / batch_count
				res[i] = P.GLB_PARAMS[P.KEY_GLB_MU] * batch_res_avg + (1 - P.GLB_PARAMS[P.KEY_GLB_MU]) * res[i]  # Exponential running average of result during epoch
				
			# Estimate dataset progress periodically
			progtracker.print_progress(count)
			if count >= total : break
			
		return res

# Autoencoding experiment on images
class AEVisionExperiment(VisionExperiment):
	def prepare_batch(self, batch):
		inputs, targets, batch_count = super().prepare_batch(batch)
		targets = {P.KEY_RECONSTR_TARGETS:inputs, P.KEY_LABEL_TARGETS: targets} # Transform targets into a dictionary containing label and reconstruction targets
		return inputs, targets, batch_count