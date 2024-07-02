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
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES, dropout_p=self.DROPOUT_P, hebb_param_dict=config.CONFIG_OPTIONS)

	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		super().set_teacher_signal(y)

class Model(nn.Module):
	# Layer names
	CONV1 = 'conv1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV_OUTPUT = BN1  # Symbolic name for the last convolutional layer providing extracted features
	FC2 = 'fc2'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	
	def __init__(self, input_shape=None, num_classes=10, dropout_p=0., hebb_param_dict=None):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.DROPOUT_P = dropout_p
		self.HEBB_PARAM_DICT = hebb_param_dict
		self.hfactory = HebbFactory(hebb_param_dict)
		
		
		# Here we define the layers of our network
		
		# First convolutional layer: 3 input channels, 96 output channels, 5x5 convolutions
		self.conv1 = self.hfactory.create_hebb_layer(in_channels=3, out_channels=96, kernel_size=5, teacher_distrib=0)
		self.bn1 = nn.BatchNorm2d(96)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = None
		self.CONV_OUTPUT_SHAPE = utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT]
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		# Final FC layer: self.CONV_OUTPUT_SHAPE-shaped input, self.NUM_CLASSES-dimensional output (one per class)
		self.fc2 = self.hfactory.create_hebb_layer(final=True, in_channels=self.CONV_OUTPUT_SHAPE[0], out_channels=self.NUM_CLASSES, kernel_size=1, teacher_distrib=1)
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		pool1_out = F.max_pool2d(conv1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SHAPE is None: return out
		
		# Linear FC layer, outputs are the class scores
		fc2_out = self.fc2(out[self.CONV_OUTPUT]).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC2] = fc2_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
	
	