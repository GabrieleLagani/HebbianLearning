import torch
import torch.nn as nn

from .. import params as P
from .. import utils


# Base class for neurolab models
class Model(nn.Module):
	def __init__(self, config, input_shape=None):
		super(Model, self).__init__()
		self.config = config
		
		# Shape of the tensors that we expect to receive as x
		self.INPUT_SHAPE = input_shape
		
		# Whether to provide a teacher signal to the deep layers
		self.DEEP_TEACHER_SIGNAL = self.config.CONFIG_OPTIONS.get(P.KEY_DEEP_TEACHER_SIGNAL, False)
	
	# Return module input shape
	def get_input_shape(self):
		return self.INPUT_SHAPE
	
	# Returns the shape of the output generated by the model for a given input shape. The queries specify which
	# output keys to take.
	def get_output_shape(self, input_shape=None, queries=None):
		if input_shape is None: input_shape = self.get_input_shape()
		if input_shape is None: return None
		training = self.training
		self.eval()
		with torch.no_grad(): # We are disabling gradient computation for processing simulated inputs
			# Generate simulated x, feed to the network, and get corresponding output
			dummy_input = {k: torch.ones(1, *s, requires_grad=False) for k, s in input_shape.items()} if isinstance(input_shape, dict) else torch.ones(1, *input_shape, requires_grad=False)
			output_shape = utils.tens2shape(self(dummy_input, queries))
		self.train(training)
		return output_shape
	
	# Specifies keys from shortcut notation. The default behavior replaces '' with default keys, '*' with all keys, a '.'
	# as separating a main key from a sub-key, and the main key itself follows the rules above. Also, '**' represents
	# all keys and subkeys, so it is interpreted as '*.**'. This default behavior can be overridden in subclasses.
	def specify_key(self, key, allkeys=None, defaults=None):
		if allkeys is None: allkeys = ['*']
		if defaults is None: defaults = ['']
		
		# If keys is None or equivalent, return default output
		if key is None or key == '': key = defaults
		
		# If keys start with '.', prepend default
		if key.startswith('.'): key = {d + key for d in defaults}
		
		# '**' is a shortcut to denote that we want to collect outputs from all layers and sub-layers
		if key == '**': key = '*.**'
		
		# '*' is a shortcut to denote that we want to collect outputs from all layers
		if key == '*': key = allkeys
		elif key.startswith('*.'): key = {a + key[1:] for a in allkeys}
		
		return set(key) if isinstance(key, (list, tuple, set)) else {key}
	
	# Here we define the flow of information through the network. The queries specify which output to take.
	def forward(self, x, queries=None, regroup=True):
		raise NotImplementedError
	
	# Function to assign specific parameter groups for optimization
	def get_param_groups(self):
		return [{'params': self.parameters()}]
	
	# Function for setting teacher signal for supervised local update rules
	def set_teacher_signal(self, y):
		for m in self.children():
			if hasattr(m, 'set_teacher_signal'): m.set_teacher_signal(y)
	
	# Function for applying local update steps
	def local_update(self):
		for m in self.children():
			if hasattr(m, 'local_update'): m.local_update()
	
	# Function to reset state of internal scheduling policies inside the model
	def reset_internal_sched_state(self):
		for m in self.children():
			if hasattr(m, 'reset_internal_sched_state'): m.reset_internal_sched_state()
	
	