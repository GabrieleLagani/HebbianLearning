import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
from neurolab.model import Model


class Net(Model):
	# Layer names
	CONV1 = 'conv1'
	RELU1 = 'relu1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV_OUTPUT = BN1 # Symbolic name for the last convolutional layer providing extracted features
	FLAT = 'flat'
	FC2 = 'fc2'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5) # 3 x channels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		
		self.CONV_OUTPUT_SIZE = utils.shape2size(utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT]))
		
		# FC Layers
		self.fc2 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_CLASSES) # conv_output_shape-dimensional x, 10-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out = F.max_pool2d(relu1_out, 2)
		bn1_out = self.bn1(pool1_out)

		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.RELU1: relu1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.CONV_OUTPUT_SIZE)
		
		# Second FC layer, outputs are the class scores
		fc2_out = self.fc2(flat)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC2] = fc2_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
