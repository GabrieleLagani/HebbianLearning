import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper

import utils


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.INPUT_SIZE = utils.shape2size(self.get_input_shape())
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(P.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		return Model(input_size=self.INPUT_SIZE, num_classes=self.NUM_CLASSES, num_hidden=self.NUM_HIDDEN, dropout_p=self.DROPOUT_P)

class Model(nn.Module):
	FLAT = 'flat'
	FC1 = 'fc1'
	RELU1 = 'relu1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	
	def __init__(self, input_size, num_classes=10, num_hidden=4096, dropout_p=0.):
		super().__init__()
		
		self.INPUT_SIZE = input_size
		self.NUM_CLASSES = num_classes
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		
		# Here we define the layers of our network
		
		# FC Layers
		self.fc1 = nn.Linear(self.INPUT_SIZE, self.NUM_HIDDEN)  # input_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn1 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.fc2 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = {}
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = x.view(-1, self.INPUT_SIZE)
		
		# Hidden Layer: FC with ReLU activations + Batch Norm
		fc1_out = self.fc1(flat)
		relu1_out = F.relu(fc1_out)
		bn1_out = self.bn1(relu1_out)
		
		# Output Layer: dropout + FC, outputs are the class scores
		fc2_out = self.fc2(F.dropout(bn1_out, p=self.DROPOUT_P, training=self.training))
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC1] = fc1_out
		out[self.RELU1] = relu1_out
		out[self.BN1] = bn1_out
		out[self.FC2] = fc2_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
