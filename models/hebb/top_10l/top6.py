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
	CONV7 = 'conv7'
	RELU7 = 'relu7'
	POOL7 = 'pool7'
	BN7 = 'bn7'
	CONV8 = 'conv8'
	RELU8 = 'relu8'
	BN8 = 'bn8'
	CONV_OUTPUT = BN8 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC9 = 'fc9'
	RELU9 = 'relu9'
	BN9 = 'bn9'
	FC10 = 'fc10'
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
		
		# Seventh convolutional layer: 256 input channels, 384 output channels, 3x3 convolutions
		self.conv7 = self.hfactory.create_hebb_layer(in_channels=256, out_channels=384, kernel_size=3, teacher_distrib=360//self.NUM_CLASSES)
		self.bn7 = nn.BatchNorm2d(384)  # Batch Norm layer
		
		# Eighth convolutional layer: 384 input channels, 512 output channels, 3x3 convolutions
		self.conv8 = self.hfactory.create_hebb_layer(in_channels=384, out_channels=512, kernel_size=3, teacher_distrib=500//self.NUM_CLASSES)
		self.bn8 = nn.BatchNorm2d(512)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = None
		self.CONV_OUTPUT_SHAPE = utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT]
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		# Ninth layer: FC, self.CONV_OUTPUT_SHAPE-shaped input, self.NUM_HIDDEN output channels
		self.fc9 = self.hfactory.create_hebb_layer(in_channels=self.CONV_OUTPUT_SHAPE[0], out_channels=self.NUM_HIDDEN, kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]), teacher_distrib=self.NUM_HIDDEN//self.NUM_CLASSES)
		self.bn9 = nn.BatchNorm2d(self.NUM_HIDDEN)  # Batch Norm layer
		
		# Final FC layer: self.NUM_HIDDEN-dimensional input, self.NUM_CLASSES-dimensional output (one per class)
		self.fc10 = self.hfactory.create_hebb_layer(final=True, in_channels=self.NUM_HIDDEN, out_channels=self.NUM_CLASSES, kernel_size=1, teacher_distrib=1)
	
	def get_conv_output(self, x):
		# Layer 7: Convolutional + 2x2 Max Pooling + Batch Norm
		conv7_out = self.conv7(x)
		pool7_out = F.max_pool2d(conv7_out, 2)
		bn7_out = HF.modified_bn(self.bn7, pool7_out)
		
		# Layer 8: Convolutional + Batch Norm
		conv8_out = self.conv8(bn7_out)
		bn8_out = HF.modified_bn(self.bn8, conv8_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV7: conv7_out,
			self.POOL7: pool7_out,
			self.BN7: bn7_out,
			self.CONV8: conv8_out,
			self.BN8: bn8_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SHAPE is None: return out
		
		# Layer 9: FC + Batch Norm
		fc9_out = self.fc9(out[self.CONV_OUTPUT])
		bn9_out = HF.modified_bn(self.bn9, fc9_out)
		
		# Linear FC layer, outputs are the class scores
		fc10_out = self.fc10(bn9_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC9] = fc9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc10_out}
		return out

