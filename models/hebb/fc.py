import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper
import hebb as H
from hebb import functional as HF
from .base import HebbFactory
import utils


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES,hebb_param_dict=config.CONFIG_OPTIONS)

	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		super().set_teacher_signal(y)

class Model(nn.Module):
	# Layer names
	FC = 'fc'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	
	def __init__(self, input_shape=None, num_classes=10, hebb_param_dict=None):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.HEBB_PARAM_DICT = hebb_param_dict
		self.hfactory = HebbFactory(hebb_param_dict)
		
		# Here we define the layers of our network
		
		# Final FC layer: input_shape-shaped input, self.NUM_CLASSES-dimensional output (one per class)
		self.fc = self.hfactory.create_hebb_layer(final=True, in_channels=input_shape[0], out_channels=self.NUM_CLASSES, kernel_size=(input_shape[1], input_shape[2]) if len(input_shape) >= 3 else 1, teacher_distrib=1)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(x if len(self.INPUT_SHAPE) >= 3 else x.view(x.size(0), x.size(1), 1, 1)).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC] = fc_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc_out}
		return out

