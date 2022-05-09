import torch
import torch.nn as nn

from .. import params as P


# Base class for network models
class Model(nn.Module):
	def __init__(self, config, input_shape=None):
		super(Model, self).__init__()
		self.config = config
		
		# Shape of the tensors that we expect to receive as x
		self.INPUT_SHAPE = input_shape if input_shape is not None else P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_INPUT_SHAPE]
	
	# Return module input shape
	def get_input_shape(self):
		return self.INPUT_SHAPE
	
	# This function forwards an input through the convolutional layers and computes the resulting output
	def get_conv_output(self, x):
		raise NotImplementedError
	
	# Feeds the network with a simulated input and compute the corresponding output feature map from any layer of a network.
	# This is useful to get the shape of network outputs (together with utils.tens2shape) in order to correctly set the
	# size of the successive layers or the input shape of successive processing stages. By default, the method considers
	# only convolutional layers. If fwd is true, all layers are considered.
	def get_dummy_fmap(self, fwd=False):
		training = self.training
		self.eval()
		with torch.no_grad(): # We are disabling gradient computation for processing simulated inputs
			# Generate simulated x, feed to the network, and get corresponding output
			dummy_input = torch.ones(1, *self.get_input_shape())
			res = (self(dummy_input) if fwd else self.get_conv_output(dummy_input))
		self.train(training)
		return res
	
	# Function for setting teacher signal for supervised local update rules
	def set_teacher_signal(self, y):
		pass
	
	# Function for applying local update steps
	def local_updates(self):
		pass
	
	# Function to assign specific parameter groups for optimization
	def get_param_groups(self):
		return [{'params': self.parameters()}]
	
	# Function to reset state of internal scheduling policies inside the model
	def reset_internal_sched_state(self):
		pass
