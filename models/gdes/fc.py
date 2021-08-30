import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
from neurolab.model import Model


class Net(Model):
	# Layer names
	FC = 'fc'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.INPUT_SIZE = utils.shape2size(self.get_input_shape())
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		# FC Layers
		self.fc = nn.Linear(self.INPUT_SIZE, self.NUM_CLASSES) # input_size-dimensional x, NUM_CLASSES-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(F.dropout(x.view(-1, self.INPUT_SIZE), p=self.DROPOUT_P, training=self.training))
		
		out[self.FC] = fc_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc_out}
		return out
