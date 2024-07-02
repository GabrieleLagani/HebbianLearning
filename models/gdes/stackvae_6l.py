import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper
from neurolab.optimization.metric import ELBOMetric
import params as PP

import utils


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_LATENT_VARS = config.CONFIG_OPTIONS.get(PP.KEY_VAE_NUM_LATENT_VARS, 256)
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(P.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		self.ELBO_BETA = config.CONFIG_OPTIONS.get(P.KEY_ELBO_BETA, 1.)
		self.ALPHA_L = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_L, 1.)
		self.ALPHA_G = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_G, 0.)
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES, num_latent=self.NUM_LATENT_VARS, num_hidden=self.NUM_HIDDEN, dropout_p=self.DROPOUT_P,
		             elbo_beta=self.ELBO_BETA, alpha_l=self.ALPHA_L, alpha_g=self.ALPHA_G)


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
	Z1 = 'z1'
	Z2 = 'z2'
	Z3 = 'z3'
	Z4 = 'z4'
	Z5 = 'z5'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	POOL_INDICES = 'pool_indices' # Name of the dictionary entry containing indices resulting from max pooling
	
	def __init__(self, input_shape=None, num_classes=10, num_latent=256, num_hidden=4096, dropout_p=0.,
		             elbo_beta=1., alpha_l=1., alpha_g=0.):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.NUM_LATENT_VARS = num_latent
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		self.ELBO_BETA = elbo_beta
		self.ALPHA_L = alpha_l
		self.ALPHA_G = alpha_g
		
		# Here we define the layers of our network and the variables to store internal gradients
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 5) # 3 input chennels, 96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		self.conv1_delta_w = torch.zeros_like(self.conv1.weight)
		self.conv1_delta_bias = torch.zeros_like(self.conv1.bias)
		self.bn1_delta_w = torch.zeros_like(self.bn1.weight)
		self.bn1_delta_bias = torch.zeros_like(self.bn1.bias)
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input chennels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		self.conv2_delta_w = torch.zeros_like(self.conv2.weight)
		self.conv2_delta_bias = torch.zeros_like(self.conv2.bias)
		self.bn2_delta_w = torch.zeros_like(self.bn2.weight)
		self.bn2_delta_bias = torch.zeros_like(self.bn2.bias)
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input chennels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		self.conv3_delta_w = torch.zeros_like(self.conv3.weight)
		self.conv3_delta_bias = torch.zeros_like(self.conv3.bias)
		self.bn3_delta_w = torch.zeros_like(self.bn3.weight)
		self.bn3_delta_bias = torch.zeros_like(self.bn3.bias)
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 256, 3)  # 192 input chennels, 256 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(256) # Batch Norm layer
		self.conv4_delta_w = torch.zeros_like(self.conv4.weight)
		self.conv4_delta_bias = torch.zeros_like(self.conv4.bias)
		self.bn4_delta_w = torch.zeros_like(self.bn4.weight)
		self.bn4_delta_bias = torch.zeros_like(self.bn4.bias)
		
		self.OUTPUT_FMAP_SHAPE = None
		self.OUTPUT_FMAP_SHAPE = {k: v for k, v in utils.get_output_fmap_shape(self, input_shape).items() if isinstance(v, torch.Tensor)}
		self.OUTPUT_FMAP_SIZE = {k: utils.shape2size(self.OUTPUT_FMAP_SHAPE[k]) for k in self.OUTPUT_FMAP_SHAPE.keys()}
		self.CONV_OUTPUT_SIZE = self.OUTPUT_FMAP_SIZE[self.CONV_OUTPUT]
		
		# FC Layers
		self.fc5 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN) # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn5 = nn.BatchNorm1d(self.NUM_HIDDEN) # Batch Norm layer
		self.fc5_delta_w = torch.zeros_like(self.fc5.weight)
		self.fc5_delta_bias = torch.zeros_like(self.fc5.bias)
		self.bn5_delta_w = torch.zeros_like(self.bn5.weight)
		self.bn5_delta_bias = torch.zeros_like(self.bn5.bias)
		self.fc6 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
		
		# Latent variable mapping layers
		self.fc_mu1 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN1], self.NUM_LATENT_VARS)  # bn1_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var1 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN1], self.NUM_LATENT_VARS)  # bn1_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu1_delta_w = torch.zeros_like(self.fc_mu1.weight)
		self.fc_mu1_delta_bias = torch.zeros_like(self.fc_mu1.bias)
		self.fc_var1_delta_w = torch.zeros_like(self.fc_var1.weight)
		self.fc_var1_delta_bias = torch.zeros_like(self.fc_var1.bias)
		self.fc_mu2 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN2], self.NUM_LATENT_VARS)  # bn2_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var2 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN2], self.NUM_LATENT_VARS)  # bn2_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu2_delta_w = torch.zeros_like(self.fc_mu2.weight)
		self.fc_mu2_delta_bias = torch.zeros_like(self.fc_mu2.bias)
		self.fc_var2_delta_w = torch.zeros_like(self.fc_var2.weight)
		self.fc_var2_delta_bias = torch.zeros_like(self.fc_var2.bias)
		self.fc_mu3 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN3], self.NUM_LATENT_VARS)  # bn3_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var3 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN3], self.NUM_LATENT_VARS)  # bn3_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu3_delta_w = torch.zeros_like(self.fc_mu3.weight)
		self.fc_mu3_delta_bias = torch.zeros_like(self.fc_mu3.bias)
		self.fc_var3_delta_w = torch.zeros_like(self.fc_var3.weight)
		self.fc_var3_delta_bias = torch.zeros_like(self.fc_var3.bias)
		self.fc_mu4 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN4], self.NUM_LATENT_VARS)  # bn4_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var4 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN4], self.NUM_LATENT_VARS)  # bn4_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu4_delta_w = torch.zeros_like(self.fc_mu4.weight)
		self.fc_mu4_delta_bias = torch.zeros_like(self.fc_mu4.bias)
		self.fc_var4_delta_w = torch.zeros_like(self.fc_var4.weight)
		self.fc_var4_delta_bias = torch.zeros_like(self.fc_var4.bias)
		self.fc_mu5 = nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var5 = nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu5_delta_w = torch.zeros_like(self.fc_mu5.weight)
		self.fc_mu5_delta_bias = torch.zeros_like(self.fc_mu5.bias)
		self.fc_var5_delta_w = torch.zeros_like(self.fc_var5.weight)
		self.fc_var5_delta_bias = torch.zeros_like(self.fc_var5.bias)
		
		# Decoding Layers
		self.dec_fc0 = nn.Linear(self.NUM_LATENT_VARS, self.NUM_HIDDEN)  # NUM_LATENT_VARS-dimensional input, self.NUM_HIDDEN-dimensional output
		self.dec_bn0 = nn.BatchNorm1d(self.NUM_HIDDEN)  # Batch Norm layer
		self.dec_fc1 = nn.Linear(self.NUM_HIDDEN, self.CONV_OUTPUT_SIZE)  # self.NUM_HIDDEN-dimensional input, CONV_OUTPUT_SIZE-dimensional output
		self.dec_fc0_delta_w = torch.zeros_like(self.dec_fc0.weight)
		self.dec_fc0_delta_bias = torch.zeros_like(self.dec_fc0.bias)
		self.dec_bn0_delta_w = torch.zeros_like(self.dec_bn0.weight)
		self.dec_bn0_delta_bias = torch.zeros_like(self.dec_bn0.bias)
		self.dec_fc1_delta_w = torch.zeros_like(self.dec_fc1.weight)
		self.dec_fc1_delta_bias = torch.zeros_like(self.dec_fc1.bias)
		self.dec_fc2 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN4])  # NUM_LATENT_VARS-dimensional input, bn4_output_size-dimensional output
		self.dec_bn2 = nn.BatchNorm2d(256)  # Batch Norm layer
		self.dec_conv2 = nn.ConvTranspose2d(256, 192, 3) # 256 input chennels, 192 output channels, 3x3 transpose convolutions
		self.dec_fc2_delta_w = torch.zeros_like(self.dec_fc2.weight)
		self.dec_fc2_delta_bias = torch.zeros_like(self.dec_fc2.bias)
		self.dec_bn2_delta_w = torch.zeros_like(self.dec_bn2.weight)
		self.dec_bn2_delta_bias = torch.zeros_like(self.dec_bn2.bias)
		self.dec_conv2_delta_w = torch.zeros_like(self.dec_conv2.weight)
		self.dec_conv2_delta_bias = torch.zeros_like(self.dec_conv2.bias)
		self.dec_fc3 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN3])  # NUM_LATENT_VARS-dimensional input, bn3_output_size-dimensional output
		self.dec_bn3 = nn.BatchNorm2d(192)  # Batch Norm layer
		self.dec_conv3 = nn.ConvTranspose2d(192, 128, 3) # 192 input chennels, 128 output channels, 3x3 transpose convolutions
		self.dec_fc3_delta_w = torch.zeros_like(self.dec_fc3.weight)
		self.dec_fc3_delta_bias = torch.zeros_like(self.dec_fc3.bias)
		self.dec_bn3_delta_w = torch.zeros_like(self.dec_bn3.weight)
		self.dec_bn3_delta_bias = torch.zeros_like(self.dec_bn3.bias)
		self.dec_conv3_delta_w = torch.zeros_like(self.dec_conv3.weight)
		self.dec_conv3_delta_bias = torch.zeros_like(self.dec_conv3.bias)
		self.dec_fc4 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN2])  # NUM_LATENT_VARS-dimensional input, bn2_output_size-dimensional output
		self.dec_bn4 = nn.BatchNorm2d(128)  # Batch Norm layer
		self.dec_conv4 = nn.ConvTranspose2d(128, 96, 3) # 128 input chennels, 96 output channels, 3x3 transpose convolutions
		self.dec_fc4_delta_w = torch.zeros_like(self.dec_fc4.weight)
		self.dec_fc4_delta_bias = torch.zeros_like(self.dec_fc4.bias)
		self.dec_bn4_delta_w = torch.zeros_like(self.dec_bn4.weight)
		self.dec_bn4_delta_bias = torch.zeros_like(self.dec_bn4.bias)
		self.dec_conv4_delta_w = torch.zeros_like(self.dec_conv4.weight)
		self.dec_conv4_delta_bias = torch.zeros_like(self.dec_conv4.bias)
		self.dec_fc5 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN1])  # NUM_LATENT_VARS-dimensional input, bn1_output_size-dimensional output
		self.dec_bn5 = nn.BatchNorm2d(96)  # Batch Norm layer
		self.dec_conv5 = nn.ConvTranspose2d(96, 3, 5) # 96 input chennels, 3 output channels, 5x5 transpose convolutions
		self.dec_fc5_delta_w = torch.zeros_like(self.dec_fc5.weight)
		self.dec_fc5_delta_bias = torch.zeros_like(self.dec_fc5.bias)
		self.dec_bn5_delta_w = torch.zeros_like(self.dec_bn5.weight)
		self.dec_bn5_delta_bias = torch.zeros_like(self.dec_bn5.bias)
		self.dec_conv5_delta_w = torch.zeros_like(self.dec_conv5.weight)
		self.dec_conv5_delta_bias = torch.zeros_like(self.dec_conv5.bias)
		
		# Internal ELBO loss function
		self.loss = ELBOMetric(self.ELBO_BETA)
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + ReLU activations + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		relu1_out = F.relu(conv1_out)
		pool1_out, pool1_indices = F.max_pool2d(relu1_out, 2, return_indices=True)
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
			self.POOL_INDICES: {
				self.POOL1: pool1_indices,
				self.POOL3: pool3_indices
			}
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		pool_indices = out[self.POOL_INDICES]
		
		if self.OUTPUT_FMAP_SHAPE is None: return out
		
		# Stretch out the feature map before feeding it to the FC layers
		flat = out[self.CONV_OUTPUT].view(-1, self.CONV_OUTPUT_SIZE)
		
		# Fifth Layer: FC with ReLU activations + Batch Norm
		fc5_out = self.fc5(flat)
		relu5_out = F.relu(fc5_out)
		bn5_out = self.bn5(relu5_out)
		
		# Sixth Layer: dropout + FC, outputs are the class scores
		fc6_out = self.fc6(F.dropout(bn5_out, p=self.DROPOUT_P, training=self.training))
		
		# Sampling
		mu1 = self.fc_mu1(out[self.BN1].view(-1, self.OUTPUT_FMAP_SIZE[self.BN1]))
		log_var1 = self.fc_var1(out[self.BN1].view(-1, self.OUTPUT_FMAP_SIZE[self.BN1]))
		std1 = torch.exp(0.5 * log_var1)
		eps1 = torch.randn_like(std1)
		z1 =  eps1 * std1 + mu1
		mu2 = self.fc_mu2(out[self.BN2].view(-1, self.OUTPUT_FMAP_SIZE[self.BN2]))
		log_var2 = self.fc_var2(out[self.BN2].view(-1, self.OUTPUT_FMAP_SIZE[self.BN2]))
		std2 = torch.exp(0.5 * log_var2)
		eps2 = torch.randn_like(std2)
		z2 =  eps2 * std2 + mu2
		mu3 = self.fc_mu3(out[self.BN3].view(-1, self.OUTPUT_FMAP_SIZE[self.BN3]))
		log_var3 = self.fc_var3(out[self.BN3].view(-1, self.OUTPUT_FMAP_SIZE[self.BN3]))
		std3 = torch.exp(0.5 * log_var3)
		eps3 = torch.randn_like(std3)
		z3 =  eps3 * std3 + mu3
		mu4 = self.fc_mu4(out[self.BN4].view(-1, self.OUTPUT_FMAP_SIZE[self.BN4]))
		log_var4 = self.fc_var4(out[self.BN4].view(-1, self.OUTPUT_FMAP_SIZE[self.BN4]))
		std4 = torch.exp(0.5 * log_var4)
		eps4 = torch.randn_like(std4)
		z4 =  eps4 * std4 + mu4
		mu5 = self.fc_mu5(bn5_out)
		log_var5 = self.fc_var5(bn5_out)
		std5 = torch.exp(0.5 * log_var5)
		eps5 = torch.randn_like(std5)
		z5 =  eps5 * std5 + mu5
		
		if self.training and self.ALPHA_L != 0: # Local updates are enabled
			# Decoding layers: FC + Batch Norm + FC or transpose convolutions
			dec_fc0_out = self.dec_fc0(z5)
			dec_relu0_out = F.relu(dec_fc0_out)
			dec_bn0_out = self.dec_bn0(dec_relu0_out)
			dec_fc1_out = self.dec_fc1(dec_bn0_out).view(-1, *self.OUTPUT_FMAP_SHAPE[self.CONV_OUTPUT])
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_fc1_out, P.KEY_ELBO_MU: mu5, P.KEY_ELBO_LOG_VAR: log_var5}, out[self.CONV_OUTPUT])
			self.fc5.zero_grad()
			self.bn5.zero_grad()
			self.fc_mu5.zero_grad()
			self.fc_var5.zero_grad()
			self.dec_fc0.zero_grad()
			self.dec_bn0.zero_grad()
			self.dec_fc1.zero_grad()
			loss.backward(retain_graph=True)
			self.fc5_delta_w = self.fc5.weight.grad.clone().detach()
			self.fc5_delta_bias = self.fc5.bias.grad.clone().detach()
			self.bn5_delta_w = self.bn5.weight.grad.clone().detach()
			self.bn5_delta_bias = self.bn5.bias.grad.clone().detach()
			self.fc_mu5_delta_w = self.fc_mu5.weight.grad.clone().detach()
			self.fc_mu5_delta_bias = self.fc_mu5.bias.grad.clone().detach()
			self.fc_var5_delta_w = self.fc_var5.weight.grad.clone().detach()
			self.fc_var5_delta_bias = self.fc_var5.bias.grad.clone().detach()
			self.dec_fc0_delta_w = self.dec_fc0.weight.grad.clone().detach()
			self.dec_fc0_delta_bias = self.dec_fc0.bias.grad.clone().detach()
			self.dec_bn0_delta_w = self.dec_bn0.weight.grad.clone().detach()
			self.dec_bn0_delta_bias = self.dec_bn0.bias.grad.clone().detach()
			self.dec_fc1_delta_w = self.dec_fc1.weight.grad.clone().detach()
			self.dec_fc1_delta_bias = self.dec_fc1.bias.grad.clone().detach()
			dec_fc2_out = self.dec_fc2(z4).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN4])
			dec_relu2_out = F.relu(dec_fc2_out)
			dec_bn2_out = self.dec_bn2(dec_relu2_out)
			dec_conv2_out = self.dec_conv2(dec_bn2_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv2_out, P.KEY_ELBO_MU: mu4, P.KEY_ELBO_LOG_VAR: log_var4}, out[self.BN3])
			self.conv4.zero_grad()
			self.bn4.zero_grad()
			self.fc_mu4.zero_grad()
			self.fc_var4.zero_grad()
			self.dec_fc2.zero_grad()
			self.dec_bn2.zero_grad()
			self.dec_conv2.zero_grad()
			loss.backward(retain_graph=True)
			self.conv4_delta_w = self.conv4.weight.grad.clone().detach()
			self.conv4_delta_bias = self.conv4.bias.grad.clone().detach()
			self.bn4_delta_w = self.bn4.weight.grad.clone().detach()
			self.bn4_delta_bias = self.bn4.bias.grad.clone().detach()
			self.fc_mu4_delta_w = self.fc_mu4.weight.grad.clone().detach()
			self.fc_mu4_delta_bias = self.fc_mu4.bias.grad.clone().detach()
			self.fc_var4_delta_w = self.fc_var4.weight.grad.clone().detach()
			self.fc_var4_delta_bias = self.fc_var4.bias.grad.clone().detach()
			self.dec_fc2_delta_w = self.dec_fc2.weight.grad.clone().detach()
			self.dec_fc2_delta_bias = self.dec_fc2.bias.grad.clone().detach()
			self.dec_bn2_delta_w = self.dec_bn2.weight.grad.clone().detach()
			self.dec_bn2_delta_bias = self.dec_bn2.bias.grad.clone().detach()
			self.dec_conv2_delta_w = self.dec_conv2.weight.grad.clone().detach()
			self.dec_conv2_delta_bias = self.dec_conv2.bias.grad.clone().detach()
			dec_fc3_out = self.dec_fc3(z3).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN3])
			dec_relu3_out = F.relu(dec_fc3_out)
			dec_pool3_out = F.max_unpool2d(dec_relu3_out, pool_indices[self.POOL3], 2)
			dec_bn3_out = self.dec_bn3(dec_pool3_out)
			dec_conv3_out = self.dec_conv3(dec_bn3_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv3_out, P.KEY_ELBO_MU: mu3, P.KEY_ELBO_LOG_VAR: log_var3}, out[self.BN2])
			self.conv3.zero_grad()
			self.bn3.zero_grad()
			self.fc_mu3.zero_grad()
			self.fc_var3.zero_grad()
			self.dec_fc3.zero_grad()
			self.dec_bn3.zero_grad()
			self.dec_conv3.zero_grad()
			loss.backward(retain_graph=True)
			self.conv3_delta_w = self.conv3.weight.grad.clone().detach()
			self.conv3_delta_bias = self.conv3.bias.grad.clone().detach()
			self.bn3_delta_w = self.bn3.weight.grad.clone().detach()
			self.bn3_delta_bias = self.bn3.bias.grad.clone().detach()
			self.fc_mu3_delta_w = self.fc_mu3.weight.grad.clone().detach()
			self.fc_mu3_delta_bias = self.fc_mu3.bias.grad.clone().detach()
			self.fc_var3_delta_w = self.fc_var3.weight.grad.clone().detach()
			self.fc_var3_delta_bias = self.fc_var3.bias.grad.clone().detach()
			self.dec_fc3_delta_w = self.dec_fc3.weight.grad.clone().detach()
			self.dec_fc3_delta_bias = self.dec_fc3.bias.grad.clone().detach()
			self.dec_bn3_delta_w = self.dec_bn3.weight.grad.clone().detach()
			self.dec_bn3_delta_bias = self.dec_bn3.bias.grad.clone().detach()
			self.dec_conv3_delta_w = self.dec_conv3.weight.grad.clone().detach()
			self.dec_conv3_delta_bias = self.dec_conv3.bias.grad.clone().detach()
			dec_fc4_out = self.dec_fc4(z2).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN2])
			dec_relu4_out = F.relu(dec_fc4_out)
			dec_bn4_out = self.dec_bn4(dec_relu4_out)
			dec_conv4_out = self.dec_conv4(dec_bn4_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv4_out, P.KEY_ELBO_MU: mu2, P.KEY_ELBO_LOG_VAR: log_var2}, out[self.BN1])
			self.conv2.zero_grad()
			self.bn2.zero_grad()
			self.fc_mu2.zero_grad()
			self.fc_var2.zero_grad()
			self.dec_fc4.zero_grad()
			self.dec_bn4.zero_grad()
			self.dec_conv4.zero_grad()
			loss.backward(retain_graph=True)
			self.conv2_delta_w = self.conv2.weight.grad.clone().detach()
			self.conv2_delta_bias = self.conv2.bias.grad.clone().detach()
			self.bn2_delta_w = self.bn2.weight.grad.clone().detach()
			self.bn2_delta_bias = self.bn2.bias.grad.clone().detach()
			self.fc_mu2_delta_w = self.fc_mu2.weight.grad.clone().detach()
			self.fc_mu2_delta_bias = self.fc_mu2.bias.grad.clone().detach()
			self.fc_var2_delta_w = self.fc_var2.weight.grad.clone().detach()
			self.fc_var2_delta_bias = self.fc_var2.bias.grad.clone().detach()
			self.dec_fc4_delta_w = self.dec_fc4.weight.grad.clone().detach()
			self.dec_fc4_delta_bias = self.dec_fc4.bias.grad.clone().detach()
			self.dec_bn4_delta_w = self.dec_bn4.weight.grad.clone().detach()
			self.dec_bn4_delta_bias = self.dec_bn4.bias.grad.clone().detach()
			self.dec_conv4_delta_w = self.dec_conv4.weight.grad.clone().detach()
			self.dec_conv4_delta_bias = self.dec_conv4.bias.grad.clone().detach()
			dec_fc5_out = self.dec_fc5(z1).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN1])
			dec_relu5_out = F.relu(dec_fc5_out)
			dec_pool5_out = F.max_unpool2d(dec_relu5_out, pool_indices[self.POOL1], 2)
			dec_bn5_out = self.dec_bn5(dec_pool5_out)
			dec_conv5_out = self.dec_conv5(dec_bn5_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv5_out, P.KEY_ELBO_MU: mu1, P.KEY_ELBO_LOG_VAR: log_var1}, x)
			self.conv1.zero_grad()
			self.bn1.zero_grad()
			self.fc_mu1.zero_grad()
			self.fc_var1.zero_grad()
			self.dec_fc5.zero_grad()
			self.dec_bn5.zero_grad()
			self.dec_conv5.zero_grad()
			loss.backward(retain_graph=True)
			self.conv1_delta_w = self.conv1.weight.grad.clone().detach()
			self.conv1_delta_bias = self.conv1.bias.grad.clone().detach()
			self.bn1_delta_w = self.bn1.weight.grad.clone().detach()
			self.bn1_delta_bias = self.bn1.bias.grad.clone().detach()
			self.fc_mu1_delta_w = self.fc_mu1.weight.grad.clone().detach()
			self.fc_mu1_delta_bias = self.fc_mu1.bias.grad.clone().detach()
			self.fc_var1_delta_w = self.fc_var1.weight.grad.clone().detach()
			self.fc_var1_delta_bias = self.fc_var1.bias.grad.clone().detach()
			self.dec_fc5_delta_w = self.dec_fc5.weight.grad.clone().detach()
			self.dec_fc5_delta_bias = self.dec_fc5.bias.grad.clone().detach()
			self.dec_bn5_delta_w = self.dec_bn5.weight.grad.clone().detach()
			self.dec_bn5_delta_bias = self.dec_bn5.bias.grad.clone().detach()
			self.dec_conv5_delta_w = self.dec_conv5.weight.grad.clone().detach()
			self.dec_conv5_delta_bias = self.dec_conv5.bias.grad.clone().detach()
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC5] = fc5_out
		out[self.RELU5] = relu5_out
		out[self.BN5] = bn5_out
		out[self.FC6] = fc6_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc6_out}
		out[self.Z1] = z1
		out[self.Z2] = z2
		out[self.Z3] = z3
		out[self.Z4] = z4
		out[self.Z5] = z5
		return out
	
	def local_update(self):
		self.fc5.weight.grad = self.ALPHA_L * self.fc5_delta_w + self.ALPHA_G * (self.fc5.weight.grad if self.fc5.weight.grad is not None else 0.)
		self.fc5.weight.bias = self.ALPHA_L * self.fc5_delta_bias + self.ALPHA_G * (self.fc5.bias.grad if self.fc5.bias.grad is not None else 0.)
		self.bn5.weight.grad = self.ALPHA_L * self.bn5_delta_w + self.ALPHA_G * (self.bn5.weight.grad if self.bn5.weight.grad is not None else 0.)
		self.bn5.weight.bias = self.ALPHA_L * self.bn5_delta_bias + self.ALPHA_G * (self.bn5.bias.grad if self.bn5.bias.grad is not None else 0.)
		self.fc_mu5.weight.grad = self.ALPHA_L * self.fc_mu5_delta_w + self.ALPHA_G * (self.fc_mu5.weight.grad if self.fc_mu5.weight.grad is not None else 0.)
		self.fc_mu5.weight.bias = self.ALPHA_L * self.fc_mu5_delta_bias + self.ALPHA_G * (self.fc_mu5.bias.grad if self.fc_mu5.bias.grad is not None else 0.)
		self.fc_var5.weight.grad = self.ALPHA_L * self.fc_var5_delta_w + self.ALPHA_G * (self.fc_var5.weight.grad if self.fc_var5.weight.grad is not None else 0.)
		self.fc_var5.weight.bias = self.ALPHA_L * self.fc_var5_delta_bias + self.ALPHA_G * (self.fc_var5.bias.grad if self.fc_var5.bias.grad is not None else 0.)
		self.dec_fc0.weight.grad = self.ALPHA_L * self.dec_fc0_delta_w + self.ALPHA_G * (self.dec_fc0.weight.grad if self.dec_fc0.weight.grad is not None else 0.)
		self.dec_fc0.weight.bias = self.ALPHA_L * self.dec_fc0_delta_bias + self.ALPHA_G * (self.dec_fc0.bias.grad if self.dec_fc0.bias.grad is not None else 0.)
		self.dec_bn0.weight.grad = self.ALPHA_L * self.dec_bn0_delta_w + self.ALPHA_G * (self.dec_bn0.weight.grad if self.dec_bn0.weight.grad is not None else 0.)
		self.dec_bn0.weight.bias = self.ALPHA_L * self.dec_bn0_delta_bias + self.ALPHA_G * (self.dec_bn0.bias.grad if self.dec_bn0.bias.grad is not None else 0.)
		self.dec_fc1.weight.grad = self.ALPHA_L * self.dec_fc1_delta_w + self.ALPHA_G * (self.dec_fc1.weight.grad if self.dec_fc1.weight.grad is not None else 0.)
		self.dec_fc1.weight.bias = self.ALPHA_L * self.dec_fc1_delta_bias + self.ALPHA_G * (self.dec_fc1.bias.grad if self.dec_fc1.bias.grad is not None else 0.)
		
		self.conv4.weight.grad = self.ALPHA_L * self.conv4_delta_w + self.ALPHA_G * (self.conv4.weight.grad if self.conv4.weight.grad is not None else 0.)
		self.conv4.weight.bias = self.ALPHA_L * self.conv4_delta_bias + self.ALPHA_G * (self.conv4.bias.grad if self.conv4.bias.grad is not None else 0.)
		self.bn4.weight.grad = self.ALPHA_L * self.bn4_delta_w + self.ALPHA_G * (self.bn4.weight.grad if self.bn4.weight.grad is not None else 0.)
		self.bn4.weight.bias = self.ALPHA_L * self.bn4_delta_bias + self.ALPHA_G * (self.bn4.bias.grad if self.bn4.bias.grad is not None else 0.)
		self.fc_mu4.weight.grad = self.ALPHA_L * self.fc_mu4_delta_w + self.ALPHA_G * (self.fc_mu4.weight.grad if self.fc_mu4.weight.grad is not None else 0.)
		self.fc_mu4.weight.bias = self.ALPHA_L * self.fc_mu4_delta_bias + self.ALPHA_G * (self.fc_mu4.bias.grad if self.fc_mu4.bias.grad is not None else 0.)
		self.fc_var4.weight.grad = self.ALPHA_L * self.fc_var4_delta_w + self.ALPHA_G * (self.fc_var4.weight.grad if self.fc_var4.weight.grad is not None else 0.)
		self.fc_var4.weight.bias = self.ALPHA_L * self.fc_var4_delta_bias + self.ALPHA_G * (self.fc_var4.bias.grad if self.fc_var4.bias.grad is not None else 0.)
		self.dec_fc2.weight.grad = self.ALPHA_L * self.dec_fc2_delta_w + self.ALPHA_G * (self.dec_fc2.weight.grad if self.dec_fc2.weight.grad is not None else 0.)
		self.dec_fc2.weight.bias = self.ALPHA_L * self.dec_fc2_delta_bias + self.ALPHA_G * (self.dec_fc2.bias.grad if self.dec_fc2.bias.grad is not None else 0.)
		self.dec_bn2.weight.grad = self.ALPHA_L * self.dec_bn2_delta_w + self.ALPHA_G * (self.dec_bn2.weight.grad if self.dec_bn2.weight.grad is not None else 0.)
		self.dec_bn2.weight.bias = self.ALPHA_L * self.dec_bn2_delta_bias + self.ALPHA_G * (self.dec_bn2.bias.grad if self.dec_bn2.bias.grad is not None else 0.)
		self.dec_conv2.weight.grad = self.ALPHA_L * self.dec_conv2_delta_w + self.ALPHA_G * (self.dec_conv2.weight.grad if self.dec_conv2.weight.grad is not None else 0.)
		self.dec_conv2.weight.bias = self.ALPHA_L * self.dec_conv2_delta_bias + self.ALPHA_G * (self.dec_conv2.bias.grad if self.dec_conv2.bias.grad is not None else 0.)
		
		self.conv3.weight.grad = self.ALPHA_L * self.conv3_delta_w + self.ALPHA_G * (self.conv3.weight.grad if self.conv3.weight.grad is not None else 0.)
		self.conv3.weight.bias = self.ALPHA_L * self.conv3_delta_bias + self.ALPHA_G * (self.conv3.bias.grad if self.conv3.bias.grad is not None else 0.)
		self.bn3.weight.grad = self.ALPHA_L * self.bn3_delta_w + self.ALPHA_G * (self.bn3.weight.grad if self.bn3.weight.grad is not None else 0.)
		self.bn3.weight.bias = self.ALPHA_L * self.bn3_delta_bias + self.ALPHA_G * (self.bn3.bias.grad if self.bn3.bias.grad is not None else 0.)
		self.fc_mu3.weight.grad = self.ALPHA_L * self.fc_mu3_delta_w + self.ALPHA_G * (self.fc_mu3.weight.grad if self.fc_mu3.weight.grad is not None else 0.)
		self.fc_mu3.weight.bias = self.ALPHA_L * self.fc_mu3_delta_bias + self.ALPHA_G * (self.fc_mu3.bias.grad if self.fc_mu3.bias.grad is not None else 0.)
		self.fc_var3.weight.grad = self.ALPHA_L * self.fc_var3_delta_w + self.ALPHA_G * (self.fc_var3.weight.grad if self.fc_var3.weight.grad is not None else 0.)
		self.fc_var3.weight.bias = self.ALPHA_L * self.fc_var3_delta_bias + self.ALPHA_G * (self.fc_var3.bias.grad if self.fc_var3.bias.grad is not None else 0.)
		self.dec_fc3.weight.grad = self.ALPHA_L * self.dec_fc3_delta_w + self.ALPHA_G * (self.dec_fc3.weight.grad if self.dec_fc3.weight.grad is not None else 0.)
		self.dec_fc3.weight.bias = self.ALPHA_L * self.dec_fc3_delta_bias + self.ALPHA_G * (self.dec_fc3.bias.grad if self.dec_fc3.bias.grad is not None else 0.)
		self.dec_bn3.weight.grad = self.ALPHA_L * self.dec_bn3_delta_w + self.ALPHA_G * (self.dec_bn3.weight.grad if self.dec_bn3.weight.grad is not None else 0.)
		self.dec_bn3.weight.bias = self.ALPHA_L * self.dec_bn3_delta_bias + self.ALPHA_G * (self.dec_bn3.bias.grad if self.dec_bn3.bias.grad is not None else 0.)
		self.dec_conv3.weight.grad = self.ALPHA_L * self.dec_conv3_delta_w + self.ALPHA_G * (self.dec_conv3.weight.grad if self.dec_conv3.weight.grad is not None else 0.)
		self.dec_conv3.weight.bias = self.ALPHA_L * self.dec_conv3_delta_bias + self.ALPHA_G * (self.dec_conv3.bias.grad if self.dec_conv3.bias.grad is not None else 0.)
		
		self.conv2.weight.grad = self.ALPHA_L * self.conv2_delta_w + self.ALPHA_G * (self.conv2.weight.grad if self.conv2.weight.grad is not None else 0.)
		self.conv2.weight.bias = self.ALPHA_L * self.conv2_delta_bias + self.ALPHA_G * (self.conv2.bias.grad if self.conv2.bias.grad is not None else 0.)
		self.bn2.weight.grad = self.ALPHA_L * self.bn2_delta_w + self.ALPHA_G * (self.bn2.weight.grad if self.bn2.weight.grad is not None else 0.)
		self.bn2.weight.bias = self.ALPHA_L * self.bn2_delta_bias + self.ALPHA_G * (self.bn2.bias.grad if self.bn2.bias.grad is not None else 0.)
		self.fc_mu2.weight.grad = self.ALPHA_L * self.fc_mu2_delta_w + self.ALPHA_G * (self.fc_mu2.weight.grad if self.fc_mu2.weight.grad is not None else 0.)
		self.fc_mu2.weight.bias = self.ALPHA_L * self.fc_mu2_delta_bias + self.ALPHA_G * (self.fc_mu2.bias.grad if self.fc_mu2.bias.grad is not None else 0.)
		self.fc_var2.weight.grad = self.ALPHA_L * self.fc_var2_delta_w + self.ALPHA_G * (self.fc_var2.weight.grad if self.fc_var2.weight.grad is not None else 0.)
		self.fc_var2.weight.bias = self.ALPHA_L * self.fc_var2_delta_bias + self.ALPHA_G * (self.fc_var2.bias.grad if self.fc_var2.bias.grad is not None else 0.)
		self.dec_fc4.weight.grad = self.ALPHA_L * self.dec_fc4_delta_w + self.ALPHA_G * (self.dec_fc4.weight.grad if self.dec_fc4.weight.grad is not None else 0.)
		self.dec_fc4.weight.bias = self.ALPHA_L * self.dec_fc4_delta_bias + self.ALPHA_G * (self.dec_fc4.bias.grad if self.dec_fc4.bias.grad is not None else 0.)
		self.dec_bn4.weight.grad = self.ALPHA_L * self.dec_bn4_delta_w + self.ALPHA_G * (self.dec_bn4.weight.grad if self.dec_bn4.weight.grad is not None else 0.)
		self.dec_bn4.weight.bias = self.ALPHA_L * self.dec_bn4_delta_bias + self.ALPHA_G * (self.dec_bn4.bias.grad if self.dec_bn4.bias.grad is not None else 0.)
		self.dec_conv4.weight.grad = self.ALPHA_L * self.dec_conv4_delta_w + self.ALPHA_G * (self.dec_conv4.weight.grad if self.dec_conv4.weight.grad is not None else 0.)
		self.dec_conv4.weight.bias = self.ALPHA_L * self.dec_conv4_delta_bias + self.ALPHA_G * (self.dec_conv4.bias.grad if self.dec_conv4.bias.grad is not None else 0.)
		
		self.conv1.weight.grad = self.ALPHA_L * self.conv1_delta_w + self.ALPHA_G * (self.conv1.weight.grad if self.conv1.weight.grad is not None else 0.)
		self.conv1.weight.bias = self.ALPHA_L * self.conv1_delta_bias + self.ALPHA_G * (self.conv1.bias.grad if self.conv1.bias.grad is not None else 0.)
		self.bn1.weight.grad = self.ALPHA_L * self.bn1_delta_w + self.ALPHA_G * (self.bn1.weight.grad if self.bn1.weight.grad is not None else 0.)
		self.bn1.weight.bias = self.ALPHA_L * self.bn1_delta_bias + self.ALPHA_G * (self.bn1.bias.grad if self.bn1.bias.grad is not None else 0.)
		self.fc_mu1.weight.grad = self.ALPHA_L * self.fc_mu1_delta_w + self.ALPHA_G * (self.fc_mu1.weight.grad if self.fc_mu1.weight.grad is not None else 0.)
		self.fc_mu1.weight.bias = self.ALPHA_L * self.fc_mu1_delta_bias + self.ALPHA_G * (self.fc_mu1.bias.grad if self.fc_mu1.bias.grad is not None else 0.)
		self.fc_var1.weight.grad = self.ALPHA_L * self.fc_var1_delta_w + self.ALPHA_G * (self.fc_var1.weight.grad if self.fc_var1.weight.grad is not None else 0.)
		self.fc_var1.weight.bias = self.ALPHA_L * self.fc_var1_delta_bias + self.ALPHA_G * (self.fc_var1.bias.grad if self.fc_var1.bias.grad is not None else 0.)
		self.dec_fc5.weight.grad = self.ALPHA_L * self.dec_fc5_delta_w + self.ALPHA_G * (self.dec_fc5.weight.grad if self.dec_fc5.weight.grad is not None else 0.)
		self.dec_fc5.weight.bias = self.ALPHA_L * self.dec_fc5_delta_bias + self.ALPHA_G * (self.dec_fc5.bias.grad if self.dec_fc5.bias.grad is not None else 0.)
		self.dec_bn5.weight.grad = self.ALPHA_L * self.dec_bn5_delta_w + self.ALPHA_G * (self.dec_bn5.weight.grad if self.dec_bn5.weight.grad is not None else 0.)
		self.dec_bn5.weight.bias = self.ALPHA_L * self.dec_bn5_delta_bias + self.ALPHA_G * (self.dec_bn5.bias.grad if self.dec_bn5.bias.grad is not None else 0.)
		self.dec_conv5.weight.grad = self.ALPHA_L * self.dec_conv5_delta_w + self.ALPHA_G * (self.dec_conv5.weight.grad if self.dec_conv5.weight.grad is not None else 0.)
		self.dec_conv5.weight.bias = self.ALPHA_L * self.dec_conv5_delta_bias + self.ALPHA_G * (self.dec_conv5.bias.grad if self.dec_conv5.bias.grad is not None else 0.)
		
			