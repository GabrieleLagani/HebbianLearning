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
	CONV5 = 'conv5'
	RELU5 = 'relu5'
	POOL5 = 'pool5'
	BN5 = 'bn5'
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
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 7) # 3 x channels, 96 output channels, 7x7 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 x channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 x channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 192, 3)  # 192 x channels, 192 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fifth convolutional layer
		self.conv5 = nn.Conv2d(192, 256, 3)  # 192 x channels, 256 output channels, 3x3 convolutions
		self.bn5 = nn.BatchNorm2d(256) # Batch Norm layer
		# Sixth convolutional layer
		self.conv6 = nn.Conv2d(256, 256, 3)  # 256 x channels, 256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256) # Batch Norm layer
		# Seventh convolutional layer
		self.conv7 = nn.Conv2d(256, 384, 3)  # 256 x channels, 384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384) # Batch Norm layer
		# Eightth convolutional layer
		self.conv8 = nn.Conv2d(384, 512, 3)  # 384 x channels, 512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512) # Batch Norm layer
		
		self.CONV_OUTPUT_SIZE = utils.shape2size(utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT]))
		
		# FC Layers
		self.fc9 = nn.Linear(self.CONV_OUTPUT_SIZE, 4096)  # conv_output_size-dimensional x, 4096-dimensional output
		self.bn9 = nn.BatchNorm1d(4096)  # Batch Norm layer
		self.fc10 = nn.Linear(4096, self.NUM_CLASSES) # 4096-dimensional x, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 3x3 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out = F.max_pool2d(relu1_out, 3)
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
		
		# Layer 5: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv5_out = self.conv5(bn4_out)
		relu5_out = F.relu(conv5_out)
		pool5_out = F.max_pool2d(relu5_out, 2)
		bn5_out = self.bn5(pool5_out)
		
		# Layer 6: Convolutional + ReLU activations + Batch Norm
		conv6_out = self.conv6(bn5_out)
		relu6_out = F.relu(conv6_out)
		bn6_out = self.bn6(relu6_out)
		
		# Layer 7: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv7_out = self.conv7(bn6_out)
		relu7_out = F.relu(conv7_out)
		pool7_out = F.max_pool2d(relu7_out, 2)
		bn7_out = self.bn7(pool7_out)
		
		# Layer 6: Convolutional + ReLU activations + Batch Norm
		conv8_out = self.conv8(bn7_out)
		relu8_out = F.relu(conv8_out)
		bn8_out = self.bn8(relu8_out)

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
			self.BN4: bn4_out,
			self.CONV5: conv5_out,
			self.RELU5: relu5_out,
			self.POOL5: pool5_out,
			self.BN5: bn5_out,
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
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc10_out}
		return out
