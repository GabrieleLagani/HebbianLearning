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
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(P.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES, num_hidden=self.NUM_HIDDEN, dropout_p=self.DROPOUT_P, hebb_param_dict=config.CONFIG_OPTIONS)
	
	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		super().set_teacher_signal(y)


class Model(nn.Module):
	# Layer names
	FC1 = 'fc1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	
	def __init__(self, input_shape=None, num_classes=10, num_hidden=4096, dropout_p=0., hebb_param_dict=None):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		self.HEBB_PARAM_DICT = hebb_param_dict
		self.hfactory = HebbFactory(hebb_param_dict)
		
		# Here we define the layers of our network
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		# First layer: FC, input_shape-shaped input, self.NUM_HIDDEN output channels
		self.fc1 = self.hfactory.create_hebb_layer(in_channels=input_shape[0], out_channels=self.NUM_HIDDEN, kernel_size=(input_shape[1], input_shape[2]) if len(input_shape) >= 3 else 1, teacher_distrib=self.NUM_HIDDEN//self.NUM_CLASSES)
		self.bn1 = nn.BatchNorm2d(self.NUM_HIDDEN)  # Batch Norm layer
		
		# self.NUM_HIDDEN-dimensional input, self.NUM_CLASSES-dimensional output (one per class)
		self.fc2 = self.hfactory.create_hebb_layer(final=True, in_channels=self.NUM_HIDDEN, out_channels=self.NUM_CLASSES, kernel_size=1, teacher_distrib=1)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Hidden Layer: FC + Batch Norm
		fc1_out = self.fc1(x if len(self.get_input_shape()) >= 3 else x.view(x.size(0), x.size(1), 1, 1))
		bn1_out = HF.modified_bn(self.bn1, fc1_out)
		
		# Output Layer, outputs are the class scores
		fc2_out = self.fc2(bn1_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC1] = fc1_out
		out[self.BN1] = bn1_out
		out[self.FC2] = fc2_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
	
	