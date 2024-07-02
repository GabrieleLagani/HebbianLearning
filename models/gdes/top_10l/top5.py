import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper

import utils


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(P.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES, num_hidden=self.NUM_HIDDEN, dropout_p=self.DROPOUT_P)


class Model(nn.Module):
	# Layer names
	CONV6 = 'conv6'
	RELU6 = 'relu6'
	BN6 = 'bn6'
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
	
	def __init__(self, input_shape=None, num_classes=10, num_hidden=4096, dropout_p=0.):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		
		# Here we define the layers of our network
		
		# Sixth convolutional layer
		self.conv6 = nn.Conv2d(256, 256, 3)  # 256 input channels, 256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256) # Batch Norm layer
		# Seventh convolutional layer
		self.conv7 = nn.Conv2d(256, 384, 3)  # 256 input channels, 384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384) # Batch Norm layer
		# Eighth convolutional layer
		self.conv8 = nn.Conv2d(384, 512, 3)  # 384 input channels, 512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512) # Batch Norm layer
		
		self.CONV_OUTPUT_SIZE = None
		self.CONV_OUTPUT_SIZE = utils.shape2size(utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT])
		
		# FC Layers
		self.fc9 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN)  # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn9 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.fc10 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 6: Convolutional + ReLU activations + Batch Norm
		conv6_out = self.conv6(x)
		relu6_out = F.relu(conv6_out)
		bn6_out = self.bn6(relu6_out)
		
		# Layer 7: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv7_out = self.conv7(bn6_out)
		relu7_out = F.relu(conv7_out)
		pool7_out = F.max_pool2d(relu7_out, 2)
		bn7_out = self.bn7(pool7_out)
		
		# Layer 8: Convolutional + ReLU activations + Batch Norm
		conv8_out = self.conv8(bn7_out)
		relu8_out = F.relu(conv8_out)
		bn8_out = self.bn8(relu8_out)

		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV6: conv6_out,
			self.RELU6: relu6_out,
			self.BN6: bn6_out,
			self.CONV7: conv7_out,
			self.RELU7: relu7_out,
			self.POOL7: pool7_out,
			self.BN7: bn7_out,
			self.CONV8: conv8_out,
			self.RELU8: relu8_out,
			self.BN8: bn8_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SIZE is None: return out
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.CONV_OUTPUT_SIZE)
		
		# Nineth Layer: FC with ReLU activations + Batch Norm
		fc9_out = self.fc9(flat)
		relu9_out = F.relu(fc9_out)
		bn9_out = self.bn9(relu9_out)
		
		# Tenth Layer: dropout + FC, outputs are the class scores
		fc10_out = self.fc10(F.dropout(bn9_out, p=self.DROPOUT_P, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC9] = fc9_out
		out[self.RELU9] = relu9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc10_out}
		return out
