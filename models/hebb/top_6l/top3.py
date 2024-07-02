import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper
import hebb as H
from hebb import functional as HF
from ..base import HebbFactory
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
	CONV4 = 'conv4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4  # Symbolic name for the last convolutional layer providing extracted features
	FC5 = 'fc5'
	BN5 = 'bn5'
	FC6 = 'fc6'
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
		
		# Fourth convolutional layer: 192 input channels, 16x16=256 output channels, 3x3 convolutions
		self.conv4 = self.hfactory.create_hebb_layer(in_channels=192, out_channels=256, kernel_size=3, teacher_distrib=240//self.NUM_CLASSES)
		self.bn4 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = None
		self.CONV_OUTPUT_SHAPE = utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT]
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		# Fifth layer: FC, self.CONV_OUTPUT_SHAPE-shaped input, self.NUM_HIDDEN output channels
		self.fc5 = self.hfactory.create_hebb_layer(in_channels=self.CONV_OUTPUT_SHAPE[0], out_channels=self.NUM_HIDDEN, kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]), teacher_distrib=self.NUM_HIDDEN//self.NUM_CLASSES)
		self.bn5 = nn.BatchNorm2d(self.NUM_HIDDEN)  # Batch Norm layer
		
		# Final FC layer: self.NUM_HIDDEN-dimensional input, self.NUM_CLASSES-dimensional output (one per class)
		self.fc6 = self.hfactory.create_hebb_layer(final=True, in_channels=self.NUM_HIDDEN, out_channels=self.NUM_CLASSES, kernel_size=1, teacher_distrib=1)
		
	def get_conv_output(self, x):
		# Layer 4: Convolutional + Batch Norm
		conv4_out = self.conv4(x)
		bn4_out = HF.modified_bn(self.bn4, conv4_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV4: conv4_out,
			self.BN4: bn4_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SHAPE is None: return out
		
		# Layer 5: FC + Batch Norm
		fc5_out = self.fc5(out[self.CONV_OUTPUT])
		bn5_out = HF.modified_bn(self.bn5, fc5_out)
		
		# Linear FC layer, outputs are the class scores
		fc6_out = self.fc6(bn5_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC5] = fc5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc6_out}
		return out

