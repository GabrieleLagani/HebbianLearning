import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
from neurolab.model import Model
import params as PP


class Net(Model):
	# Layer names
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
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(PP.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		# Here we define the layers of our network
		
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(self.get_input_shape()[0], 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		
		self.CONV_OUTPUT_SIZE = utils.shape2size(utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT]))
		
		# FC Layers
		self.fc5 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN)  # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn5 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.fc6 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 4: Convolutional + ReLU activations + Batch Norm
		conv4_out = self.conv4(x)
		relu4_out = F.relu(conv4_out)
		bn4_out = self.bn4(relu4_out)

		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV4: conv4_out,
			self.RELU4: relu4_out,
			self.BN4: bn4_out
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
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
