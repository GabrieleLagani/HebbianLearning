import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
from neurolab.model import Model
import params as PP


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
	Z = 'z'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	VAE_OUTPUT = 'vae_output' # Name for the vae output consisting of reconstruction and latent variables statistics
	POOL_INDICES = 'pool_indices' # Name of the dictionary entry containing indices resulting from max pooling
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(PP.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		self.NUM_LATENT_VARS = config.CONFIG_OPTIONS.get(PP.KEY_VAE_NUM_LATENT_VARS, 256)
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 7) # 3 input channels, 96 output channels, 7x7 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 192, 3)  # 192 input channels, 192 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(192) # Batch Norm layer
		# Fifth convolutional layer
		self.conv5 = nn.Conv2d(192, 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn5 = nn.BatchNorm2d(256) # Batch Norm layer
		# Sixth convolutional layer
		self.conv6 = nn.Conv2d(256, 256, 3)  # 256 input channels, 256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256) # Batch Norm layer
		# Seventh convolutional layer
		self.conv7 = nn.Conv2d(256, 384, 3)  # 256 input channels, 384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384) # Batch Norm layer
		# Eighth convolutional layer
		self.conv8 = nn.Conv2d(384, 512, 3)  # 384 input channels, 512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512) # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT])
		self.CONV_OUTPUT_SIZE = utils.shape2size(self.CONV_OUTPUT_SHAPE)
		
		# FC Layers
		self.fc9 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN)  # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn9 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.fc10 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
		self.fc_mu =  nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var =  nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		
		# Decoding Layers
		self.dec_fc0 = nn.Linear(self.NUM_LATENT_VARS, self.NUM_HIDDEN)  # NUM_LATENT_VARS-dimensional input, self.NUM_HIDDEN-dimensional output
		self.dec_bn0 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.dec_fc1 = nn.Linear(self.NUM_HIDDEN, self.CONV_OUTPUT_SIZE)  # self.NUM_HIDDEN-dimensional input, CONV_OUTPUT_SIZE-dimensional output
		self.dec_bn1 = nn.BatchNorm1d(self.CONV_OUTPUT_SIZE)  # Batch Norm layer
		self.dec_conv2 = nn.ConvTranspose2d(512, 384, 3) # 512 input channels, 384 output channels, 3x3 transpose convolutions
		self.dec_bn2 = nn.BatchNorm2d(384) # Batch Norm layer
		self.dec_conv3 = nn.ConvTranspose2d(384, 256, 3) # 384 input channels, 256 output channels, 3x3 transpose convolutions
		self.dec_bn3 = nn.BatchNorm2d(256) # Batch Norm layer
		self.dec_conv4 = nn.ConvTranspose2d(256, 256, 3) # 256 input channels, 256 output channels, 3x3 transpose convolutions
		self.dec_bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		self.dec_conv5 = nn.ConvTranspose2d(256, 192, 3) # 256 input channels, 192 output channels, 3x3 transpose convolutions
		self.dec_bn5 = nn.BatchNorm2d(192) # Batch Norm layer
		self.dec_conv6 = nn.ConvTranspose2d(192, 192, 3) # 192 input channels, 192 output channels, 3x3 transpose convolutions
		self.dec_bn6 = nn.BatchNorm2d(192) # Batch Norm layer
		self.dec_conv7 = nn.ConvTranspose2d(192, 128, 3) # 192 input channels, 128 output channels, 3x3 transpose convolutions
		self.dec_bn7 = nn.BatchNorm2d(128) # Batch Norm layer
		self.dec_conv8 = nn.ConvTranspose2d(128, 96, 3) # 128 input channels, 96 output channels, 3x3 transpose convolutions
		self.dec_bn8 = nn.BatchNorm2d(96) # Batch Norm layer
		self.dec_conv9 = nn.ConvTranspose2d(96, 3, 7) # 96 input channels, 3 output channels, 7x7 transpose convolutions
		self.dec_bn9 = nn.BatchNorm2d(3) # Batch Norm layer
		
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 3x3 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out, pool1_indices = F.max_pool2d(relu1_out, 3, return_indices=True)
		bn1_out = self.bn1(pool1_out)
		
		# Layer 2: Convolutional + ReLU activations + Batch Norm
		conv2_out = self.conv2(bn1_out)
		relu2_out = F.relu(conv2_out)
		bn2_out = self.bn2(relu2_out)
		
		# Layer 3: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv3_out = self.conv3(bn2_out)
		relu3_out = F.relu(conv3_out)
		pool3_out, pool3_indices = F.max_pool2d(relu3_out, 2, return_indices=True)
		bn3_out = self.bn3(pool3_out)
		
		# Layer 4: Convolutional + ReLU activations + Batch Norm
		conv4_out = self.conv4(bn3_out)
		relu4_out = F.relu(conv4_out)
		bn4_out = self.bn4(relu4_out)
		
		# Layer 5: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv5_out = self.conv5(bn4_out)
		relu5_out = F.relu(conv5_out)
		pool5_out, pool5_indices = F.max_pool2d(relu5_out, 2, return_indices=True)
		bn5_out = self.bn5(pool5_out)
		
		# Layer 6: Convolutional + ReLU activations + Batch Norm
		conv6_out = self.conv6(bn5_out)
		relu6_out = F.relu(conv6_out)
		bn6_out = self.bn6(relu6_out)
		
		# Layer 7: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv7_out = self.conv7(bn6_out)
		relu7_out = F.relu(conv7_out)
		pool7_out, pool7_indices = F.max_pool2d(relu7_out, 2, return_indices=True)
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
			self.POOL_INDICES: {
				self.POOL1: pool1_indices,
				self.POOL3: pool3_indices,
				self.POOL5: pool5_indices,
				self.POOL7: pool7_indices
			}
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		pool_indices = out[self.POOL_INDICES]
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.CONV_OUTPUT_SIZE)
		
		# Nineth Layer: FC with ReLU activations + Batch Norm
		fc9_out = self.fc9(flat)
		relu9_out = F.relu(fc9_out)
		bn9_out = self.bn9(relu9_out)
		
		# Tenth Layer: dropout + FC, outputs are the class scores
		fc10_out = self.fc10(F.dropout(bn9_out, p=self.DROPOUT_P, training=self.training))
		
		# Sampling
		mu = self.fc_mu(bn9_out)
		log_var = self.fc_var(bn9_out)
		std = torch.exp(0.5 * log_var)
		eps = torch.randn_like(std)
		z =  eps * std + mu
		
		# Decoding layers: double FC + transpose convolutions + Batch Norm
		dec_fc0_out = self.dec_fc0(z)
		dec_relu0_out = F.relu(dec_fc0_out)
		dec_bn0_out = self.dec_bn0(dec_relu0_out)
		dec_fc1_out = self.dec_fc1(dec_bn0_out)
		dec_relu1_out = F.relu(dec_fc1_out)
		dec_bn1_out = self.dec_bn1(dec_relu1_out)
		dec_conv2_out = self.dec_conv2(dec_bn1_out.view(-1, *self.CONV_OUTPUT_SHAPE))
		dec_relu2_out = F.relu(dec_conv2_out)
		dec_pool2_out = F.max_unpool2d(dec_relu2_out, pool_indices[self.POOL7], 2)
		dec_bn2_out = self.dec_bn2(dec_pool2_out)
		dec_conv3_out = self.dec_conv3(dec_bn2_out)
		dec_relu3_out = F.relu(dec_conv3_out)
		dec_bn3_out = self.dec_bn3(dec_relu3_out)
		dec_conv4_out = self.dec_conv4(dec_bn3_out)
		dec_relu4_out = F.relu(dec_conv4_out)
		dec_pool4_out = F.max_unpool2d(dec_relu4_out, pool_indices[self.POOL5], 2)
		dec_bn4_out = self.dec_bn4(dec_pool4_out)
		dec_conv5_out = self.dec_conv5(dec_bn4_out)
		dec_relu5_out = F.relu(dec_conv5_out)
		dec_bn5_out = self.dec_bn5(dec_relu5_out)
		dec_conv6_out = self.dec_conv6(dec_bn5_out)
		dec_relu6_out = F.relu(dec_conv6_out)
		dec_pool6_out = F.max_unpool2d(dec_relu6_out, pool_indices[self.POOL3], 2)
		dec_bn6_out = self.dec_bn6(dec_pool6_out)
		dec_conv7_out = self.dec_conv7(dec_bn6_out)
		dec_relu7_out = F.relu(dec_conv7_out)
		dec_bn7_out = self.dec_bn7(dec_relu7_out)
		dec_conv8_out = self.dec_conv8(dec_bn7_out)
		dec_relu8_out = F.relu(dec_conv8_out)
		dec_pool8_out = F.max_unpool2d(dec_relu8_out, pool_indices[self.POOL1], 3)
		dec_bn8_out = self.dec_bn8(dec_pool8_out)
		dec_conv9_out = self.dec_conv9(dec_bn8_out)
		dec_bn9_out = self.dec_bn9(dec_conv9_out)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC9] = fc9_out
		out[self.RELU9] = relu9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		out[self.Z] = z
		out[self.VAE_OUTPUT] = {
			P.KEY_CLASS_SCORES: fc10_out,
			P.KEY_AUTOENC_RECONSTR: dec_bn9_out,
			P.KEY_ELBO_MU: mu,
			P.KEY_ELBO_LOG_VAR: log_var}
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc10_out}
		return out
