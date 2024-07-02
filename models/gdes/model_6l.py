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
	CONV1 = 'conv1'
	RELU1 = 'relu1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV2 = 'conv2'
	RELU2 = 'relu2'
	BN2 = 'bn2'
	CONV3 = 'conv3'
	RELU3 = 'relu3'
	POOL3 = 'pool3'
	BN3 = 'bn3'
	CONV4 = 'conv4'
	RELU4 = 'relu4'
	BN4 = 'bn4'
	CONV_OUTPUT = BN4 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC5 = 'fc5'
	RELU5 = 'relu5'
	BN5 = 'bn5'
	FC6 = 'fc6'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, input_shape=None, num_classes=10, num_hidden=4096, dropout_p=0.):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5) # 3 input channels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		
		self.CONV_OUTPUT_SIZE = None
		self.CONV_OUTPUT_SIZE = utils.shape2size(utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT])
		
		# FC Layers
		self.fc5 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN)  # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn5 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.fc6 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, 10-dimensional output (one per class)
	
	def get_output_key(self):
		return self.CLASS_SCORES
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out = F.max_pool2d(relu1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Layer 2: Convolutional + ReLU activations + Batch Norm
		conv2_out = self.conv2(bn1_out)
		relu2_out = F.relu(conv2_out)
		bn2_out = self.bn2(relu2_out)
		
		# Layer 3: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv3_out = self.conv3(bn2_out)
		relu3_out = F.relu(conv3_out)
		pool3_out = F.max_pool2d(relu3_out, 2)
		bn3_out = self.bn3(pool3_out)
		
		# Layer 4: Convolutional + ReLU activations + Batch Norm
		conv4_out = self.conv4(bn3_out)
		relu4_out = F.relu(conv4_out)
		bn4_out = self.bn4(relu4_out)

		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.RELU1: relu1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
			self.CONV2: conv2_out,
			self.RELU2: relu2_out,
			self.BN2: bn2_out,
			self.CONV3: conv3_out,
			self.RELU3: relu3_out,
			self.POOL3: pool3_out,
			self.BN3: bn3_out,
			self.CONV4: conv4_out,
			self.RELU4: relu4_out,
			self.BN4: bn4_out
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SIZE is None: return out
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.CONV_OUTPUT_SIZE)
		
		# Fifth Layer: FC with ReLU activations + Batch Norm
		fc5_out = self.fc5(flat)
		relu5_out = F.relu(fc5_out)
		bn5_out = self.bn5(relu5_out)
		
		# Sixth Layer: dropout + FC, outputs are the class scores
		fc6_out = self.fc6(F.dropout(bn5_out, p=self.DROPOUT_P, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC5] = fc5_out
		out[self.RELU5] = relu5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc6_out}
		return out

