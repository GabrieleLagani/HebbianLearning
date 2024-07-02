import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper

import utils


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.INPUT_SIZE = utils.shape2size(self.get_input_shape())
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		return Model(input_size=self.INPUT_SIZE, num_classes=self.NUM_CLASSES, dropout_p=self.DROPOUT_P)

	
class Model(nn.Module):
	# Layer names
	FC = 'fc'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, input_size, num_classes=10, dropout_p=0.):
		super().__init__()
		
		self.INPUT_SIZE = input_size
		self.NUM_CLASSES = num_classes
		self.DROPOUT_P = dropout_p
		
		# FC Layers
		self.fc = nn.Linear(self.INPUT_SIZE, self.NUM_CLASSES) # input_size-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(F.dropout(x.view(-1, self.INPUT_SIZE), p=self.DROPOUT_P, training=self.training))
		
		out[self.FC] = fc_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc_out}
		return out
