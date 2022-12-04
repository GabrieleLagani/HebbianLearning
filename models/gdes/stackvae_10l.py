import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
from neurolab.model import Model
from neurolab.optimization.metric import ELBOMetric
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
	Z1 = 'z1'
	Z2 = 'z2'
	Z3 = 'z3'
	Z4 = 'z4'
	Z5 = 'z5'
	Z6 = 'z6'
	Z7 = 'z7'
	Z8 = 'z8'
	Z9 = 'z9'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	VAE_OUTPUT = 'vae_output' # Name for the vae output consisting of reconstruction and latent variables statistics
	POOL_INDICES = 'pool_indices' # Name of the dictionary entry containing indices resulting from max pooling
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(PP.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		self.NUM_LATENT_VARS = config.CONFIG_OPTIONS.get(PP.KEY_VAE_NUM_LATENT_VARS, 256)
		self.ELBO_BETA = config.CONFIG_OPTIONS.get(P.KEY_ELBO_BETA, 1.)
		self.ALPHA_L = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_L, 1.)
		self.ALPHA_G = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_G, 0.)
		
		# Here we define the layers of our network and the variables to store internal gradients
		
		# First convolutional layer
		self.conv1 = nn.Conv2d(3, 96, 7) # 3 input channels, 96 output channels, 7x7 convolutions
		self.bn1 = nn.BatchNorm2d(96) # Batch Norm layer
		self.conv1_delta_w = torch.zeros_like(self.conv1.weight)
		self.conv1_delta_bias = torch.zeros_like(self.conv1.bias)
		self.bn1_delta_w = torch.zeros_like(self.bn1.weight)
		self.bn1_delta_bias = torch.zeros_like(self.bn1.bias)
		# Second convolutional layer
		self.conv2 = nn.Conv2d(96, 128, 3) # 96 input channels, 128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128) # Batch Norm layer
		self.conv2_delta_w = torch.zeros_like(self.conv2.weight)
		self.conv2_delta_bias = torch.zeros_like(self.conv2.bias)
		self.bn2_delta_w = torch.zeros_like(self.bn2.weight)
		self.bn2_delta_bias = torch.zeros_like(self.bn2.bias)
		# Third convolutional layer
		self.conv3 = nn.Conv2d(128, 192, 3)  # 128 input channels, 192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192) # Batch Norm layer
		self.conv3_delta_w = torch.zeros_like(self.conv3.weight)
		self.conv3_delta_bias = torch.zeros_like(self.conv3.bias)
		self.bn3_delta_w = torch.zeros_like(self.bn3.weight)
		self.bn3_delta_bias = torch.zeros_like(self.bn3.bias)
		# Fourth convolutional layer
		self.conv4 = nn.Conv2d(192, 192, 3)  # 192 input channels, 192 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(192) # Batch Norm layer
		self.conv4_delta_w = torch.zeros_like(self.conv4.weight)
		self.conv4_delta_bias = torch.zeros_like(self.conv4.bias)
		self.bn4_delta_w = torch.zeros_like(self.bn4.weight)
		self.bn4_delta_bias = torch.zeros_like(self.bn4.bias)
		# Fifth convolutional layer
		self.conv5 = nn.Conv2d(192, 256, 3)  # 192 input channels, 256 output channels, 3x3 convolutions
		self.bn5 = nn.BatchNorm2d(256) # Batch Norm layer
		self.conv5_delta_w = torch.zeros_like(self.conv5.weight)
		self.conv5_delta_bias = torch.zeros_like(self.conv5.bias)
		self.bn5_delta_w = torch.zeros_like(self.bn5.weight)
		self.bn5_delta_bias = torch.zeros_like(self.bn5.bias)
		# Sixth convolutional layer
		self.conv6 = nn.Conv2d(256, 256, 3)  # 256 input channels, 256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256) # Batch Norm layer
		self.conv6_delta_w = torch.zeros_like(self.conv6.weight)
		self.conv6_delta_bias = torch.zeros_like(self.conv6.bias)
		self.bn6_delta_w = torch.zeros_like(self.bn6.weight)
		self.bn6_delta_bias = torch.zeros_like(self.bn6.bias)
		# Seventh convolutional layer
		self.conv7 = nn.Conv2d(256, 384, 3)  # 256 input channels, 384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384) # Batch Norm layer
		self.conv7_delta_w = torch.zeros_like(self.conv7.weight)
		self.conv7_delta_bias = torch.zeros_like(self.conv7.bias)
		self.bn7_delta_w = torch.zeros_like(self.bn7.weight)
		self.bn7_delta_bias = torch.zeros_like(self.bn7.bias)
		# Eightth convolutional layer
		self.conv8 = nn.Conv2d(384, 512, 3)  # 384 input channels, 512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512) # Batch Norm layer
		self.conv8_delta_w = torch.zeros_like(self.conv8.weight)
		self.conv8_delta_bias = torch.zeros_like(self.conv8.bias)
		self.bn8_delta_w = torch.zeros_like(self.bn8.weight)
		self.bn8_delta_bias = torch.zeros_like(self.bn8.bias)
		
		self.OUTPUT_FMAP_SHAPE = {k: utils.tens2shape(v) for k, v in self.get_dummy_fmap().items() if isinstance(v, torch.Tensor)}
		self.OUTPUT_FMAP_SIZE = {k: utils.shape2size(self.OUTPUT_FMAP_SHAPE[k]) for k in self.OUTPUT_FMAP_SHAPE.keys()}
		self.CONV_OUTPUT_SIZE = self.OUTPUT_FMAP_SIZE[self.CONV_OUTPUT]
		
		# FC Layers
		self.fc9 = nn.Linear(self.CONV_OUTPUT_SIZE, self.NUM_HIDDEN) # conv_output_size-dimensional input, self.NUM_HIDDEN-dimensional output
		self.bn9 = nn.BatchNorm1d(self.NUM_HIDDEN) # Batch Norm layer
		self.fc9_delta_w = torch.zeros_like(self.fc9.weight)
		self.fc9_delta_bias = torch.zeros_like(self.fc9.bias)
		self.bn9_delta_w = torch.zeros_like(self.bn9.weight)
		self.bn9_delta_bias = torch.zeros_like(self.bn9.bias)
		self.fc10 = nn.Linear(self.NUM_HIDDEN, self.NUM_CLASSES) # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
		
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
		self.fc_mu5 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN5], self.NUM_LATENT_VARS)  # bn5_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var5 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN5], self.NUM_LATENT_VARS)  # bn5_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu5_delta_w = torch.zeros_like(self.fc_mu5.weight)
		self.fc_mu5_delta_bias = torch.zeros_like(self.fc_mu5.bias)
		self.fc_var5_delta_w = torch.zeros_like(self.fc_var5.weight)
		self.fc_var5_delta_bias = torch.zeros_like(self.fc_var5.bias)
		self.fc_mu6 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN6], self.NUM_LATENT_VARS)  # bn6_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var6 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN6], self.NUM_LATENT_VARS)  # bn6_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu6_delta_w = torch.zeros_like(self.fc_mu6.weight)
		self.fc_mu6_delta_bias = torch.zeros_like(self.fc_mu6.bias)
		self.fc_var6_delta_w = torch.zeros_like(self.fc_var6.weight)
		self.fc_var6_delta_bias = torch.zeros_like(self.fc_var6.bias)
		self.fc_mu7 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN7], self.NUM_LATENT_VARS)  # bn7_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var7 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN7], self.NUM_LATENT_VARS)  # bn7_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu7_delta_w = torch.zeros_like(self.fc_mu7.weight)
		self.fc_mu7_delta_bias = torch.zeros_like(self.fc_mu7.bias)
		self.fc_var7_delta_w = torch.zeros_like(self.fc_var7.weight)
		self.fc_var7_delta_bias = torch.zeros_like(self.fc_var7.bias)
		self.fc_mu8 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN8], self.NUM_LATENT_VARS)  # bn8_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var8 = nn.Linear(self.OUTPUT_FMAP_SIZE[self.BN8], self.NUM_LATENT_VARS)  # bn8_output_size-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu8_delta_w = torch.zeros_like(self.fc_mu8.weight)
		self.fc_mu8_delta_bias = torch.zeros_like(self.fc_mu8.bias)
		self.fc_var8_delta_w = torch.zeros_like(self.fc_var8.weight)
		self.fc_var8_delta_bias = torch.zeros_like(self.fc_var8.bias)
		self.fc_mu9 = nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_var9 = nn.Linear(self.NUM_HIDDEN, self.NUM_LATENT_VARS)  # self.NUM_HIDDEN-dimensional input, NUM_LATENT_VARS-dimensional output
		self.fc_mu9_delta_w = torch.zeros_like(self.fc_mu9.weight)
		self.fc_mu9_delta_bias = torch.zeros_like(self.fc_mu9.bias)
		self.fc_var9_delta_w = torch.zeros_like(self.fc_var9.weight)
		self.fc_var9_delta_bias = torch.zeros_like(self.fc_var9.bias)
		
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
		self.dec_fc2 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN8])  # NUM_LATENT_VARS-dimensional input, bn8_output_size-dimensional output
		self.dec_bn2 = nn.BatchNorm2d(512)  # Batch Norm layer
		self.dec_conv2 = nn.ConvTranspose2d(512, 384, 3) # 512 input chennels, 384 output channels, 3x3 transpose convolutions
		self.dec_fc2_delta_w = torch.zeros_like(self.dec_fc2.weight)
		self.dec_fc2_delta_bias = torch.zeros_like(self.dec_fc2.bias)
		self.dec_bn2_delta_w = torch.zeros_like(self.dec_bn2.weight)
		self.dec_bn2_delta_bias = torch.zeros_like(self.dec_bn2.bias)
		self.dec_conv2_delta_w = torch.zeros_like(self.dec_conv2.weight)
		self.dec_conv2_delta_bias = torch.zeros_like(self.dec_conv2.bias)
		self.dec_fc3 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN7])  # NUM_LATENT_VARS-dimensional input, bn7_output_size-dimensional output
		self.dec_bn3 = nn.BatchNorm2d(384)  # Batch Norm layer
		self.dec_conv3 = nn.ConvTranspose2d(384, 256, 3) # 384 input chennels, 256 output channels, 3x3 transpose convolutions
		self.dec_fc3_delta_w = torch.zeros_like(self.dec_fc3.weight)
		self.dec_fc3_delta_bias = torch.zeros_like(self.dec_fc3.bias)
		self.dec_bn3_delta_w = torch.zeros_like(self.dec_bn3.weight)
		self.dec_bn3_delta_bias = torch.zeros_like(self.dec_bn3.bias)
		self.dec_conv3_delta_w = torch.zeros_like(self.dec_conv3.weight)
		self.dec_conv3_delta_bias = torch.zeros_like(self.dec_conv3.bias)
		self.dec_fc4 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN6])  # NUM_LATENT_VARS-dimensional input, bn6_output_size-dimensional output
		self.dec_bn4 = nn.BatchNorm2d(256)  # Batch Norm layer
		self.dec_conv4 = nn.ConvTranspose2d(256, 256, 3) # 256 input chennels, 256 output channels, 3x3 transpose convolutions
		self.dec_fc4_delta_w = torch.zeros_like(self.dec_fc4.weight)
		self.dec_fc4_delta_bias = torch.zeros_like(self.dec_fc4.bias)
		self.dec_bn4_delta_w = torch.zeros_like(self.dec_bn4.weight)
		self.dec_bn4_delta_bias = torch.zeros_like(self.dec_bn4.bias)
		self.dec_conv4_delta_w = torch.zeros_like(self.dec_conv4.weight)
		self.dec_conv4_delta_bias = torch.zeros_like(self.dec_conv4.bias)
		self.dec_fc5 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN5])  # NUM_LATENT_VARS-dimensional input, bn5_output_size-dimensional output
		self.dec_bn5 = nn.BatchNorm2d(256)  # Batch Norm layer
		self.dec_conv5 = nn.ConvTranspose2d(256, 192, 3) # 256 input chennels, 192 output channels, 3x3 transpose convolutions
		self.dec_fc5_delta_w = torch.zeros_like(self.dec_fc5.weight)
		self.dec_fc5_delta_bias = torch.zeros_like(self.dec_fc5.bias)
		self.dec_bn5_delta_w = torch.zeros_like(self.dec_bn5.weight)
		self.dec_bn5_delta_bias = torch.zeros_like(self.dec_bn5.bias)
		self.dec_conv5_delta_w = torch.zeros_like(self.dec_conv5.weight)
		self.dec_conv5_delta_bias = torch.zeros_like(self.dec_conv5.bias)
		self.dec_fc6 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN4])  # NUM_LATENT_VARS-dimensional input, bn4_output_size-dimensional output
		self.dec_bn6 = nn.BatchNorm2d(192)  # Batch Norm layer
		self.dec_conv6 = nn.ConvTranspose2d(192, 192, 3) # 192 input chennels, 192 output channels, 3x3 transpose convolutions
		self.dec_fc6_delta_w = torch.zeros_like(self.dec_fc6.weight)
		self.dec_fc6_delta_bias = torch.zeros_like(self.dec_fc6.bias)
		self.dec_bn6_delta_w = torch.zeros_like(self.dec_bn6.weight)
		self.dec_bn6_delta_bias = torch.zeros_like(self.dec_bn6.bias)
		self.dec_conv6_delta_w = torch.zeros_like(self.dec_conv6.weight)
		self.dec_conv6_delta_bias = torch.zeros_like(self.dec_conv6.bias)
		self.dec_fc7 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN3])  # NUM_LATENT_VARS-dimensional input, bn3_output_size-dimensional output
		self.dec_bn7 = nn.BatchNorm2d(192)  # Batch Norm layer
		self.dec_conv7 = nn.ConvTranspose2d(192, 128, 3) # 192 input chennels, 128 output channels, 3x3 transpose convolutions
		self.dec_fc7_delta_w = torch.zeros_like(self.dec_fc7.weight)
		self.dec_fc7_delta_bias = torch.zeros_like(self.dec_fc7.bias)
		self.dec_bn7_delta_w = torch.zeros_like(self.dec_bn7.weight)
		self.dec_bn7_delta_bias = torch.zeros_like(self.dec_bn7.bias)
		self.dec_conv7_delta_w = torch.zeros_like(self.dec_conv7.weight)
		self.dec_conv7_delta_bias = torch.zeros_like(self.dec_conv7.bias)
		self.dec_fc8 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN2])  # NUM_LATENT_VARS-dimensional input, bn2_output_size-dimensional output
		self.dec_bn8 = nn.BatchNorm2d(128)  # Batch Norm layer
		self.dec_conv8 = nn.ConvTranspose2d(128, 96, 3) # 128 input chennels, 96 output channels, 3x3 transpose convolutions
		self.dec_fc8_delta_w = torch.zeros_like(self.dec_fc8.weight)
		self.dec_fc8_delta_bias = torch.zeros_like(self.dec_fc8.bias)
		self.dec_bn8_delta_w = torch.zeros_like(self.dec_bn8.weight)
		self.dec_bn8_delta_bias = torch.zeros_like(self.dec_bn8.bias)
		self.dec_conv8_delta_w = torch.zeros_like(self.dec_conv8.weight)
		self.dec_conv8_delta_bias = torch.zeros_like(self.dec_conv8.bias)
		self.dec_fc9 = nn.Linear(self.NUM_LATENT_VARS, self.OUTPUT_FMAP_SIZE[self.BN1])  # NUM_LATENT_VARS-dimensional input, bn1_output_size-dimensional output
		self.dec_bn9 = nn.BatchNorm2d(96)  # Batch Norm layer
		self.dec_conv9 = nn.ConvTranspose2d(96, 3, 7) # 96 input chennels, 3 output channels, 7x7 transpose convolutions
		self.dec_fc9_delta_w = torch.zeros_like(self.dec_fc9.weight)
		self.dec_fc9_delta_bias = torch.zeros_like(self.dec_fc9.bias)
		self.dec_bn9_delta_w = torch.zeros_like(self.dec_bn9.weight)
		self.dec_bn9_delta_bias = torch.zeros_like(self.dec_bn9.bias)
		self.dec_conv9_delta_w = torch.zeros_like(self.dec_conv9.weight)
		self.dec_conv9_delta_bias = torch.zeros_like(self.dec_conv9.bias)
		
		# Internal ELBO loss function
		self.loss = ELBOMetric(self.ELBO_BETA)
	
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
		mu4 = self.fc_mu4(out[self.BN4.view(-1, self.OUTPUT_FMAP_SIZE[self.BN4])])
		log_var4 = self.fc_var4(out[self.BN4].view(-1, self.OUTPUT_FMAP_SIZE[self.BN4]))
		std4 = torch.exp(0.5 * log_var4)
		eps4 = torch.randn_like(std4)
		z4 =  eps4 * std4 + mu4
		mu5 = self.fc_mu5(out[self.BN5].view(-1, self.OUTPUT_FMAP_SIZE[self.BN5]))
		log_var5 = self.fc_var5(out[self.BN5].view(-1, self.OUTPUT_FMAP_SIZE[self.BN5]))
		std5 = torch.exp(0.5 * log_var5)
		eps5 = torch.randn_like(std5)
		z5 =  eps5 * std5 + mu5
		mu6 = self.fc_mu6(out[self.BN6].view(-1, self.OUTPUT_FMAP_SIZE[self.BN6]))
		log_var6 = self.fc_var6(out[self.BN6].view(-1, self.OUTPUT_FMAP_SIZE[self.BN6]))
		std6 = torch.exp(0.5 * log_var6)
		eps6 = torch.randn_like(std6)
		z6 =  eps6 * std6 + mu6
		mu7 = self.fc_mu7(out[self.BN7].view(-1, self.OUTPUT_FMAP_SIZE[self.BN7]))
		log_var7 = self.fc_var7(out[self.BN7].view(-1, self.OUTPUT_FMAP_SIZE[self.BN7]))
		std7 = torch.exp(0.5 * log_var7)
		eps7 = torch.randn_like(std7)
		z7 =  eps7 * std7 + mu7
		mu8 = self.fc_mu8(out[self.BN8].view(-1, self.OUTPUT_FMAP_SIZE[self.BN8]))
		log_var8 = self.fc_var8(out[self.BN8].view(-1, self.OUTPUT_FMAP_SIZE[self.BN8]))
		std8 = torch.exp(0.5 * log_var8)
		eps8 = torch.randn_like(std8)
		z8 =  eps8 * std8 + mu8
		mu9 = self.fc_mu9(bn9_out)
		log_var9 = self.fc_var9(bn9_out)
		std9 = torch.exp(0.5 * log_var9)
		eps9 = torch.randn_like(std9)
		z9 =  eps9 * std9 + mu9
		
		if self.training and self.ALPHA_L != 0: # Local updates are enabled
			# Decoding layers: FC + Batch Norm + FC or transpose convolutions
			dec_fc0_out = self.dec_fc0(z9)
			dec_relu0_out = F.relu(dec_fc0_out)
			dec_bn0_out = self.dec_bn0(dec_relu0_out)
			dec_fc1_out = self.dec_fc1(dec_bn0_out).view(-1, *self.OUTPUT_FMAP_SHAPE[self.CONV_OUTPUT])
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_fc1_out, P.KEY_ELBO_MU: mu9, P.KEY_ELBO_LOG_VAR: log_var9}, out[self.CONV_OUTPUT])
			self.fc9.zero_grad()
			self.bn9.zero_grad()
			self.fc_mu9.zero_grad()
			self.fc_var9.zero_grad()
			self.dec_fc0.zero_grad()
			self.dec_bn0.zero_grad()
			self.dec_fc1.zero_grad()
			loss.backward(retain_graph=True)
			self.fc9_delta_w = self.fc9.weight.grad.clone().detach()
			self.fc9_delta_bias = self.fc9.bias.grad.clone().detach()
			self.bn9_delta_w = self.bn9.weight.grad.clone().detach()
			self.bn9_delta_bias = self.bn9.bias.grad.clone().detach()
			self.fc_mu9_delta_w = self.fc_mu9.weight.grad.clone().detach()
			self.fc_mu9_delta_bias = self.fc_mu9.bias.grad.clone().detach()
			self.fc_var9_delta_w = self.fc_var9.weight.grad.clone().detach()
			self.fc_var9_delta_bias = self.fc_var9.bias.grad.clone().detach()
			self.dec_fc0_delta_w = self.dec_fc0.weight.grad.clone().detach()
			self.dec_fc0_delta_bias = self.dec_fc0.bias.grad.clone().detach()
			self.dec_bn0_delta_w = self.dec_bn0.weight.grad.clone().detach()
			self.dec_bn0_delta_bias = self.dec_bn0.bias.grad.clone().detach()
			self.dec_fc1_delta_w = self.dec_fc1.weight.grad.clone().detach()
			self.dec_fc1_delta_bias = self.dec_fc1.bias.grad.clone().detach()
			dec_fc2_out = self.dec_fc2(z8).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN8])
			dec_relu2_out = F.relu(dec_fc2_out)
			dec_bn2_out = self.dec_bn2(dec_relu2_out)
			dec_conv2_out = self.dec_conv2(dec_bn2_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv2_out, P.KEY_ELBO_MU: mu8, P.KEY_ELBO_LOG_VAR: log_var8}, out[self.BN7])
			self.conv8.zero_grad()
			self.bn8.zero_grad()
			self.fc_mu8.zero_grad()
			self.fc_var8.zero_grad()
			self.dec_fc2.zero_grad()
			self.dec_bn2.zero_grad()
			self.dec_conv2.zero_grad()
			loss.backward(retain_graph=True)
			self.conv8_delta_w = self.conv8.weight.grad.clone().detach()
			self.conv8_delta_bias = self.conv8.bias.grad.clone().detach()
			self.bn8_delta_w = self.bn8.weight.grad.clone().detach()
			self.bn8_delta_bias = self.bn8.bias.grad.clone().detach()
			self.fc_mu8_delta_w = self.fc_mu8.weight.grad.clone().detach()
			self.fc_mu8_delta_bias = self.fc_mu8.bias.grad.clone().detach()
			self.fc_var8_delta_w = self.fc_var8.weight.grad.clone().detach()
			self.fc_var8_delta_bias = self.fc_var8.bias.grad.clone().detach()
			self.dec_fc2_delta_w = self.dec_fc2.weight.grad.clone().detach()
			self.dec_fc2_delta_bias = self.dec_fc2.bias.grad.clone().detach()
			self.dec_bn2_delta_w = self.dec_bn2.weight.grad.clone().detach()
			self.dec_bn2_delta_bias = self.dec_bn2.bias.grad.clone().detach()
			self.dec_conv2_delta_w = self.dec_conv2.weight.grad.clone().detach()
			self.dec_conv2_delta_bias = self.dec_conv2.bias.grad.clone().detach()
			dec_fc3_out = self.dec_fc3(z7).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN7])
			dec_relu3_out = F.relu(dec_fc3_out)
			dec_pool3_out = F.max_unpool2d(dec_relu3_out, pool_indices[self.POOL7], 2)
			dec_bn3_out = self.dec_bn3(dec_pool3_out)
			dec_conv3_out = self.dec_conv3(dec_bn3_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv3_out, P.KEY_ELBO_MU: mu7, P.KEY_ELBO_LOG_VAR: log_var7}, out[self.BN6])
			self.conv7.zero_grad()
			self.bn7.zero_grad()
			self.fc_mu7.zero_grad()
			self.fc_var7.zero_grad()
			self.dec_fc3.zero_grad()
			self.dec_bn3.zero_grad()
			self.dec_conv3.zero_grad()
			loss.backward(retain_graph=True)
			self.conv7_delta_w = self.conv7.weight.grad.clone().detach()
			self.conv7_delta_bias = self.conv7.bias.grad.clone().detach()
			self.bn7_delta_w = self.bn7.weight.grad.clone().detach()
			self.bn7_delta_bias = self.bn7.bias.grad.clone().detach()
			self.fc_mu7_delta_w = self.fc_mu7.weight.grad.clone().detach()
			self.fc_mu7_delta_bias = self.fc_mu7.bias.grad.clone().detach()
			self.fc_var7_delta_w = self.fc_var7.weight.grad.clone().detach()
			self.fc_var7_delta_bias = self.fc_var7.bias.grad.clone().detach()
			self.dec_fc3_delta_w = self.dec_fc3.weight.grad.clone().detach()
			self.dec_fc3_delta_bias = self.dec_fc3.bias.grad.clone().detach()
			self.dec_bn3_delta_w = self.dec_bn3.weight.grad.clone().detach()
			self.dec_bn3_delta_bias = self.dec_bn3.bias.grad.clone().detach()
			self.dec_conv3_delta_w = self.dec_conv3.weight.grad.clone().detach()
			self.dec_conv3_delta_bias = self.dec_conv3.bias.grad.clone().detach()
			dec_fc4_out = self.dec_fc4(z6).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN6])
			dec_relu4_out = F.relu(dec_fc4_out)
			dec_bn4_out = self.dec_bn4(dec_relu4_out)
			dec_conv4_out = self.dec_conv4(dec_bn4_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv4_out, P.KEY_ELBO_MU: mu6, P.KEY_ELBO_LOG_VAR: log_var6}, out[self.BN5])
			self.conv6.zero_grad()
			self.bn6.zero_grad()
			self.fc_mu6.zero_grad()
			self.fc_var6.zero_grad()
			self.dec_fc4.zero_grad()
			self.dec_bn4.zero_grad()
			self.dec_conv4.zero_grad()
			loss.backward(retain_graph=True)
			self.conv6_delta_w = self.conv6.weight.grad.clone().detach()
			self.conv6_delta_bias = self.conv6.bias.grad.clone().detach()
			self.bn6_delta_w = self.bn6.weight.grad.clone().detach()
			self.bn6_delta_bias = self.bn6.bias.grad.clone().detach()
			self.fc_mu6_delta_w = self.fc_mu6.weight.grad.clone().detach()
			self.fc_mu6_delta_bias = self.fc_mu6.bias.grad.clone().detach()
			self.fc_var6_delta_w = self.fc_var6.weight.grad.clone().detach()
			self.fc_var6_delta_bias = self.fc_var6.bias.grad.clone().detach()
			self.dec_fc4_delta_w = self.dec_fc4.weight.grad.clone().detach()
			self.dec_fc4_delta_bias = self.dec_fc4.bias.grad.clone().detach()
			self.dec_bn4_delta_w = self.dec_bn4.weight.grad.clone().detach()
			self.dec_bn4_delta_bias = self.dec_bn4.bias.grad.clone().detach()
			self.dec_conv4_delta_w = self.dec_conv4.weight.grad.clone().detach()
			self.dec_conv4_delta_bias = self.dec_conv4.bias.grad.clone().detach()
			dec_fc5_out = self.dec_fc5(z5).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN5])
			dec_relu5_out = F.relu(dec_fc5_out)
			dec_pool5_out = F.max_unpool2d(dec_relu5_out, pool_indices[self.POOL5], 2)
			dec_bn5_out = self.dec_bn5(dec_pool5_out)
			dec_conv5_out = self.dec_conv5(dec_bn5_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv5_out, P.KEY_ELBO_MU: mu5, P.KEY_ELBO_LOG_VAR: log_var5}, out[self.BN4])
			self.conv5.zero_grad()
			self.bn5.zero_grad()
			self.fc_mu5.zero_grad()
			self.fc_var5.zero_grad()
			self.dec_fc5.zero_grad()
			self.dec_bn5.zero_grad()
			self.dec_conv5.zero_grad()
			loss.backward(retain_graph=True)
			self.conv5_delta_w = self.conv5.weight.grad.clone().detach()
			self.conv5_delta_bias = self.conv5.bias.grad.clone().detach()
			self.bn5_delta_w = self.bn5.weight.grad.clone().detach()
			self.bn5_delta_bias = self.bn5.bias.grad.clone().detach()
			self.fc_mu5_delta_w = self.fc_mu5.weight.grad.clone().detach()
			self.fc_mu5_delta_bias = self.fc_mu5.bias.grad.clone().detach()
			self.fc_var5_delta_w = self.fc_var5.weight.grad.clone().detach()
			self.fc_var5_delta_bias = self.fc_var5.bias.grad.clone().detach()
			self.dec_fc5_delta_w = self.dec_fc5.weight.grad.clone().detach()
			self.dec_fc5_delta_bias = self.dec_fc5.bias.grad.clone().detach()
			self.dec_bn5_delta_w = self.dec_bn5.weight.grad.clone().detach()
			self.dec_bn5_delta_bias = self.dec_bn5.bias.grad.clone().detach()
			self.dec_conv5_delta_w = self.dec_conv5.weight.grad.clone().detach()
			self.dec_conv5_delta_bias = self.dec_conv5.bias.grad.clone().detach()
			dec_fc6_out = self.dec_fc6(z4).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN4])
			dec_relu6_out = F.relu(dec_fc6_out)
			dec_bn6_out = self.dec_bn6(dec_relu6_out)
			dec_conv6_out = self.dec_conv6(dec_bn6_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv6_out, P.KEY_ELBO_MU: mu4, P.KEY_ELBO_LOG_VAR: log_var4}, out[self.BN3])
			self.conv4.zero_grad()
			self.bn4.zero_grad()
			self.fc_mu4.zero_grad()
			self.fc_var4.zero_grad()
			self.dec_fc6.zero_grad()
			self.dec_bn6.zero_grad()
			self.dec_conv6.zero_grad()
			loss.backward(retain_graph=True)
			self.conv4_delta_w = self.conv4.weight.grad.clone().detach()
			self.conv4_delta_bias = self.conv4.bias.grad.clone().detach()
			self.bn4_delta_w = self.bn4.weight.grad.clone().detach()
			self.bn4_delta_bias = self.bn4.bias.grad.clone().detach()
			self.fc_mu4_delta_w = self.fc_mu4.weight.grad.clone().detach()
			self.fc_mu4_delta_bias = self.fc_mu4.bias.grad.clone().detach()
			self.fc_var4_delta_w = self.fc_var4.weight.grad.clone().detach()
			self.fc_var4_delta_bias = self.fc_var4.bias.grad.clone().detach()
			self.dec_fc6_delta_w = self.dec_fc6.weight.grad.clone().detach()
			self.dec_fc6_delta_bias = self.dec_fc6.bias.grad.clone().detach()
			self.dec_bn6_delta_w = self.dec_bn6.weight.grad.clone().detach()
			self.dec_bn6_delta_bias = self.dec_bn6.bias.grad.clone().detach()
			self.dec_conv6_delta_w = self.dec_conv6.weight.grad.clone().detach()
			self.dec_conv6_delta_bias = self.dec_conv6.bias.grad.clone().detach()
			dec_fc7_out = self.dec_fc7(z3).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN3])
			dec_relu7_out = F.relu(dec_fc7_out)
			dec_pool7_out = F.max_unpool2d(dec_relu7_out, pool_indices[self.POOL3], 2)
			dec_bn7_out = self.dec_bn7(dec_pool7_out)
			dec_conv7_out = self.dec_conv7(dec_bn7_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv7_out, P.KEY_ELBO_MU: mu3, P.KEY_ELBO_LOG_VAR: log_var3}, out[self.BN2])
			self.conv3.zero_grad()
			self.bn3.zero_grad()
			self.fc_mu3.zero_grad()
			self.fc_var3.zero_grad()
			self.dec_fc7.zero_grad()
			self.dec_bn7.zero_grad()
			self.dec_conv7.zero_grad()
			loss.backward(retain_graph=True)
			self.conv3_delta_w = self.conv3.weight.grad.clone().detach()
			self.conv3_delta_bias = self.conv3.bias.grad.clone().detach()
			self.bn3_delta_w = self.bn3.weight.grad.clone().detach()
			self.bn3_delta_bias = self.bn3.bias.grad.clone().detach()
			self.fc_mu3_delta_w = self.fc_mu3.weight.grad.clone().detach()
			self.fc_mu3_delta_bias = self.fc_mu3.bias.grad.clone().detach()
			self.fc_var3_delta_w = self.fc_var3.weight.grad.clone().detach()
			self.fc_var3_delta_bias = self.fc_var3.bias.grad.clone().detach()
			self.dec_fc7_delta_w = self.dec_fc7.weight.grad.clone().detach()
			self.dec_fc7_delta_bias = self.dec_fc7.bias.grad.clone().detach()
			self.dec_bn7_delta_w = self.dec_bn7.weight.grad.clone().detach()
			self.dec_bn7_delta_bias = self.dec_bn7.bias.grad.clone().detach()
			self.dec_conv7_delta_w = self.dec_conv7.weight.grad.clone().detach()
			self.dec_conv7_delta_bias = self.dec_conv7.bias.grad.clone().detach()
			dec_fc8_out = self.dec_fc8(z2).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN2])
			dec_relu8_out = F.relu(dec_fc8_out)
			dec_bn8_out = self.dec_bn8(dec_relu8_out)
			dec_conv8_out = self.dec_conv8(dec_bn8_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv8_out, P.KEY_ELBO_MU: mu2, P.KEY_ELBO_LOG_VAR: log_var2}, out[self.BN1])
			self.conv2.zero_grad()
			self.bn2.zero_grad()
			self.fc_mu2.zero_grad()
			self.fc_var2.zero_grad()
			self.dec_fc8.zero_grad()
			self.dec_bn8.zero_grad()
			self.dec_conv8.zero_grad()
			loss.backward(retain_graph=True)
			self.conv2_delta_w = self.conv2.weight.grad.clone().detach()
			self.conv2_delta_bias = self.conv2.bias.grad.clone().detach()
			self.bn2_delta_w = self.bn2.weight.grad.clone().detach()
			self.bn2_delta_bias = self.bn2.bias.grad.clone().detach()
			self.fc_mu2_delta_w = self.fc_mu2.weight.grad.clone().detach()
			self.fc_mu2_delta_bias = self.fc_mu2.bias.grad.clone().detach()
			self.fc_var2_delta_w = self.fc_var2.weight.grad.clone().detach()
			self.fc_var2_delta_bias = self.fc_var2.bias.grad.clone().detach()
			self.dec_fc8_delta_w = self.dec_fc8.weight.grad.clone().detach()
			self.dec_fc8_delta_bias = self.dec_fc8.bias.grad.clone().detach()
			self.dec_bn8_delta_w = self.dec_bn8.weight.grad.clone().detach()
			self.dec_bn8_delta_bias = self.dec_bn8.bias.grad.clone().detach()
			self.dec_conv8_delta_w = self.dec_conv8.weight.grad.clone().detach()
			self.dec_conv8_delta_bias = self.dec_conv8.bias.grad.clone().detach()
			dec_fc9_out = self.dec_fc9(z1).view(-1, *self.OUTPUT_FMAP_SHAPE[self.BN1])
			dec_relu9_out = F.relu(dec_fc9_out)
			dec_pool9_out = F.max_unpool2d(dec_relu9_out, pool_indices[self.POOL1], 3)
			dec_bn9_out = self.dec_bn9(dec_pool9_out)
			dec_conv9_out = self.dec_conv9(dec_bn9_out)
			loss = self.loss({P.KEY_AUTOENC_RECONSTR: dec_conv9_out, P.KEY_ELBO_MU: mu1, P.KEY_ELBO_LOG_VAR: log_var1}, x)
			self.conv1.zero_grad()
			self.bn1.zero_grad()
			self.fc_mu1.zero_grad()
			self.fc_var1.zero_grad()
			self.dec_fc9.zero_grad()
			self.dec_bn9.zero_grad()
			self.dec_conv9.zero_grad()
			loss.backward(retain_graph=True)
			self.conv1_delta_w = self.conv1.weight.grad.clone().detach()
			self.conv1_delta_bias = self.conv1.bias.grad.clone().detach()
			self.bn1_delta_w = self.bn1.weight.grad.clone().detach()
			self.bn1_delta_bias = self.bn1.bias.grad.clone().detach()
			self.fc_mu1_delta_w = self.fc_mu1.weight.grad.clone().detach()
			self.fc_mu1_delta_bias = self.fc_mu1.bias.grad.clone().detach()
			self.fc_var1_delta_w = self.fc_var1.weight.grad.clone().detach()
			self.fc_var1_delta_bias = self.fc_var1.bias.grad.clone().detach()
			self.dec_fc9_delta_w = self.dec_fc9.weight.grad.clone().detach()
			self.dec_fc9_delta_bias = self.dec_fc9.bias.grad.clone().detach()
			self.dec_bn9_delta_w = self.dec_bn9.weight.grad.clone().detach()
			self.dec_bn9_delta_bias = self.dec_bn9.bias.grad.clone().detach()
			self.dec_conv9_delta_w = self.dec_conv9.weight.grad.clone().detach()
			self.dec_conv9_delta_bias = self.dec_conv9.bias.grad.clone().detach()
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FLAT] = flat
		out[self.FC9] = fc9_out
		out[self.RELU9] = relu9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		out[self.Z1] = z1
		out[self.Z2] = z2
		out[self.Z3] = z3
		out[self.Z4] = z4
		out[self.Z5] = z5
		out[self.Z6] = z6
		out[self.Z7] = z7
		out[self.Z8] = z8
		out[self.Z9] = z9
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc10_out}
		return out
	
	def local_updates(self):
		self.fc9.weight.grad = self.ALPHA_L * self.fc9_delta_w + self.ALPHA_G * (self.fc9.weight.grad if self.fc9.weight.grad is not None else 0.)
		self.fc9.weight.bias = self.ALPHA_L * self.fc9_delta_bias + self.ALPHA_G * (self.fc9.bias.grad if self.fc9.bias.grad is not None else 0.)
		self.bn9.weight.grad = self.ALPHA_L * self.bn9_delta_w + self.ALPHA_G * (self.bn9.weight.grad if self.bn9.weight.grad is not None else 0.)
		self.bn9.weight.bias = self.ALPHA_L * self.bn9_delta_bias + self.ALPHA_G * (self.bn9.bias.grad if self.bn9.bias.grad is not None else 0.)
		self.fc_mu9.weight.grad = self.ALPHA_L * self.fc_mu9_delta_w + self.ALPHA_G * (self.fc_mu9.weight.grad if self.fc_mu9.weight.grad is not None else 0.)
		self.fc_mu9.weight.bias = self.ALPHA_L * self.fc_mu9_delta_bias + self.ALPHA_G * (self.fc_mu9.bias.grad if self.fc_mu9.bias.grad is not None else 0.)
		self.fc_var9.weight.grad = self.ALPHA_L * self.fc_var9_delta_w + self.ALPHA_G * (self.fc_var9.weight.grad if self.fc_var9.weight.grad is not None else 0.)
		self.fc_var9.weight.bias = self.ALPHA_L * self.fc_var9_delta_bias + self.ALPHA_G * (self.fc_var9.bias.grad if self.fc_var9.bias.grad is not None else 0.)
		self.dec_fc0.weight.grad = self.ALPHA_L * self.dec_fc0_delta_w + self.ALPHA_G * (self.dec_fc0.weight.grad if self.dec_fc0.weight.grad is not None else 0.)
		self.dec_fc0.weight.bias = self.ALPHA_L * self.dec_fc0_delta_bias + self.ALPHA_G * (self.dec_fc0.bias.grad if self.dec_fc0.bias.grad is not None else 0.)
		self.dec_bn0.weight.grad = self.ALPHA_L * self.dec_bn0_delta_w + self.ALPHA_G * (self.dec_bn0.weight.grad if self.dec_bn0.weight.grad is not None else 0.)
		self.dec_bn0.weight.bias = self.ALPHA_L * self.dec_bn0_delta_bias + self.ALPHA_G * (self.dec_bn0.bias.grad if self.dec_bn0.bias.grad is not None else 0.)
		self.dec_fc1.weight.grad = self.ALPHA_L * self.dec_fc1_delta_w + self.ALPHA_G * (self.dec_fc1.weight.grad if self.dec_fc1.weight.grad is not None else 0.)
		self.dec_fc1.weight.bias = self.ALPHA_L * self.dec_fc1_delta_bias + self.ALPHA_G * (self.dec_fc1.bias.grad if self.dec_fc1.bias.grad is not None else 0.)
		
		self.conv8.weight.grad = self.ALPHA_L * self.conv8_delta_w + self.ALPHA_G * (self.conv8.weight.grad if self.conv8.weight.grad is not None else 0.)
		self.conv8.weight.bias = self.ALPHA_L * self.conv8_delta_bias + self.ALPHA_G * (self.conv8.bias.grad if self.conv8.bias.grad is not None else 0.)
		self.bn8.weight.grad = self.ALPHA_L * self.bn8_delta_w + self.ALPHA_G * (self.bn8.weight.grad if self.bn8.weight.grad is not None else 0.)
		self.bn8.weight.bias = self.ALPHA_L * self.bn8_delta_bias + self.ALPHA_G * (self.bn8.bias.grad if self.bn8.bias.grad is not None else 0.)
		self.fc_mu8.weight.grad = self.ALPHA_L * self.fc_mu8_delta_w + self.ALPHA_G * (self.fc_mu8.weight.grad if self.fc_mu8.weight.grad is not None else 0.)
		self.fc_mu8.weight.bias = self.ALPHA_L * self.fc_mu8_delta_bias + self.ALPHA_G * (self.fc_mu8.bias.grad if self.fc_mu8.bias.grad is not None else 0.)
		self.fc_var8.weight.grad = self.ALPHA_L * self.fc_var8_delta_w + self.ALPHA_G * (self.fc_var8.weight.grad if self.fc_var8.weight.grad is not None else 0.)
		self.fc_var8.weight.bias = self.ALPHA_L * self.fc_var8_delta_bias + self.ALPHA_G * (self.fc_var8.bias.grad if self.fc_var8.bias.grad is not None else 0.)
		self.dec_fc2.weight.grad = self.ALPHA_L * self.dec_fc2_delta_w + self.ALPHA_G * (self.dec_fc2.weight.grad if self.dec_fc2.weight.grad is not None else 0.)
		self.dec_fc2.weight.bias = self.ALPHA_L * self.dec_fc2_delta_bias + self.ALPHA_G * (self.dec_fc2.bias.grad if self.dec_fc2.bias.grad is not None else 0.)
		self.dec_bn2.weight.grad = self.ALPHA_L * self.dec_bn2_delta_w + self.ALPHA_G * (self.dec_bn2.weight.grad if self.dec_bn2.weight.grad is not None else 0.)
		self.dec_bn2.weight.bias = self.ALPHA_L * self.dec_bn2_delta_bias + self.ALPHA_G * (self.dec_bn2.bias.grad if self.dec_bn2.bias.grad is not None else 0.)
		self.dec_conv2.weight.grad = self.ALPHA_L * self.dec_conv2_delta_w + self.ALPHA_G * (self.dec_conv2.weight.grad if self.dec_conv2.weight.grad is not None else 0.)
		self.dec_conv2.weight.bias = self.ALPHA_L * self.dec_conv2_delta_bias + self.ALPHA_G * (self.dec_conv2.bias.grad if self.dec_conv2.bias.grad is not None else 0.)
		
		self.conv7.weight.grad = self.ALPHA_L * self.conv7_delta_w + self.ALPHA_G * (self.conv7.weight.grad if self.conv7.weight.grad is not None else 0.)
		self.conv7.weight.bias = self.ALPHA_L * self.conv7_delta_bias + self.ALPHA_G * (self.conv7.bias.grad if self.conv7.bias.grad is not None else 0.)
		self.bn7.weight.grad = self.ALPHA_L * self.bn7_delta_w + self.ALPHA_G * (self.bn7.weight.grad if self.bn7.weight.grad is not None else 0.)
		self.bn7.weight.bias = self.ALPHA_L * self.bn7_delta_bias + self.ALPHA_G * (self.bn7.bias.grad if self.bn7.bias.grad is not None else 0.)
		self.fc_mu7.weight.grad = self.ALPHA_L * self.fc_mu7_delta_w + self.ALPHA_G * (self.fc_mu7.weight.grad if self.fc_mu7.weight.grad is not None else 0.)
		self.fc_mu7.weight.bias = self.ALPHA_L * self.fc_mu7_delta_bias + self.ALPHA_G * (self.fc_mu7.bias.grad if self.fc_mu7.bias.grad is not None else 0.)
		self.fc_var7.weight.grad = self.ALPHA_L * self.fc_var7_delta_w + self.ALPHA_G * (self.fc_var7.weight.grad if self.fc_var7.weight.grad is not None else 0.)
		self.fc_var7.weight.bias = self.ALPHA_L * self.fc_var7_delta_bias + self.ALPHA_G * (self.fc_var7.bias.grad if self.fc_var7.bias.grad is not None else 0.)
		self.dec_fc3.weight.grad = self.ALPHA_L * self.dec_fc3_delta_w + self.ALPHA_G * (self.dec_fc3.weight.grad if self.dec_fc3.weight.grad is not None else 0.)
		self.dec_fc3.weight.bias = self.ALPHA_L * self.dec_fc3_delta_bias + self.ALPHA_G * (self.dec_fc3.bias.grad if self.dec_fc3.bias.grad is not None else 0.)
		self.dec_bn3.weight.grad = self.ALPHA_L * self.dec_bn3_delta_w + self.ALPHA_G * (self.dec_bn3.weight.grad if self.dec_bn3.weight.grad is not None else 0.)
		self.dec_bn3.weight.bias = self.ALPHA_L * self.dec_bn3_delta_bias + self.ALPHA_G * (self.dec_bn3.bias.grad if self.dec_bn3.bias.grad is not None else 0.)
		self.dec_conv3.weight.grad = self.ALPHA_L * self.dec_conv3_delta_w + self.ALPHA_G * (self.dec_conv3.weight.grad if self.dec_conv3.weight.grad is not None else 0.)
		self.dec_conv3.weight.bias = self.ALPHA_L * self.dec_conv3_delta_bias + self.ALPHA_G * (self.dec_conv3.bias.grad if self.dec_conv3.bias.grad is not None else 0.)
		
		self.conv6.weight.grad = self.ALPHA_L * self.conv6_delta_w + self.ALPHA_G * (self.conv6.weight.grad if self.conv6.weight.grad is not None else 0.)
		self.conv6.weight.bias = self.ALPHA_L * self.conv6_delta_bias + self.ALPHA_G * (self.conv6.bias.grad if self.conv6.bias.grad is not None else 0.)
		self.bn6.weight.grad = self.ALPHA_L * self.bn6_delta_w + self.ALPHA_G * (self.bn6.weight.grad if self.bn6.weight.grad is not None else 0.)
		self.bn6.weight.bias = self.ALPHA_L * self.bn6_delta_bias + self.ALPHA_G * (self.bn6.bias.grad if self.bn6.bias.grad is not None else 0.)
		self.fc_mu6.weight.grad = self.ALPHA_L * self.fc_mu6_delta_w + self.ALPHA_G * (self.fc_mu6.weight.grad if self.fc_mu6.weight.grad is not None else 0.)
		self.fc_mu6.weight.bias = self.ALPHA_L * self.fc_mu6_delta_bias + self.ALPHA_G * (self.fc_mu6.bias.grad if self.fc_mu6.bias.grad is not None else 0.)
		self.fc_var6.weight.grad = self.ALPHA_L * self.fc_var6_delta_w + self.ALPHA_G * (self.fc_var6.weight.grad if self.fc_var6.weight.grad is not None else 0.)
		self.fc_var6.weight.bias = self.ALPHA_L * self.fc_var6_delta_bias + self.ALPHA_G * (self.fc_var6.bias.grad if self.fc_var6.bias.grad is not None else 0.)
		self.dec_fc4.weight.grad = self.ALPHA_L * self.dec_fc4_delta_w + self.ALPHA_G * (self.dec_fc4.weight.grad if self.dec_fc4.weight.grad is not None else 0.)
		self.dec_fc4.weight.bias = self.ALPHA_L * self.dec_fc4_delta_bias + self.ALPHA_G * (self.dec_fc4.bias.grad if self.dec_fc4.bias.grad is not None else 0.)
		self.dec_bn4.weight.grad = self.ALPHA_L * self.dec_bn4_delta_w + self.ALPHA_G * (self.dec_bn4.weight.grad if self.dec_bn4.weight.grad is not None else 0.)
		self.dec_bn4.weight.bias = self.ALPHA_L * self.dec_bn4_delta_bias + self.ALPHA_G * (self.dec_bn4.bias.grad if self.dec_bn4.bias.grad is not None else 0.)
		self.dec_conv4.weight.grad = self.ALPHA_L * self.dec_conv4_delta_w + self.ALPHA_G * (self.dec_conv4.weight.grad if self.dec_conv4.weight.grad is not None else 0.)
		self.dec_conv4.weight.bias = self.ALPHA_L * self.dec_conv4_delta_bias + self.ALPHA_G * (self.dec_conv4.bias.grad if self.dec_conv4.bias.grad is not None else 0.)
		
		self.conv5.weight.grad = self.ALPHA_L * self.conv5_delta_w + self.ALPHA_G * (self.conv5.weight.grad if self.conv5.weight.grad is not None else 0.)
		self.conv5.weight.bias = self.ALPHA_L * self.conv5_delta_bias + self.ALPHA_G * (self.conv5.bias.grad if self.conv5.bias.grad is not None else 0.)
		self.bn5.weight.grad = self.ALPHA_L * self.bn5_delta_w + self.ALPHA_G * (self.bn5.weight.grad if self.bn5.weight.grad is not None else 0.)
		self.bn5.weight.bias = self.ALPHA_L * self.bn5_delta_bias + self.ALPHA_G * (self.bn5.bias.grad if self.bn5.bias.grad is not None else 0.)
		self.fc_mu5.weight.grad = self.ALPHA_L * self.fc_mu5_delta_w + self.ALPHA_G * (self.fc_mu5.weight.grad if self.fc_mu5.weight.grad is not None else 0.)
		self.fc_mu5.weight.bias = self.ALPHA_L * self.fc_mu5_delta_bias + self.ALPHA_G * (self.fc_mu5.bias.grad if self.fc_mu5.bias.grad is not None else 0.)
		self.fc_var5.weight.grad = self.ALPHA_L * self.fc_var5_delta_w + self.ALPHA_G * (self.fc_var5.weight.grad if self.fc_var5.weight.grad is not None else 0.)
		self.fc_var5.weight.bias = self.ALPHA_L * self.fc_var5_delta_bias + self.ALPHA_G * (self.fc_var5.bias.grad if self.fc_var5.bias.grad is not None else 0.)
		self.dec_fc5.weight.grad = self.ALPHA_L * self.dec_fc5_delta_w + self.ALPHA_G * (self.dec_fc5.weight.grad if self.dec_fc5.weight.grad is not None else 0.)
		self.dec_fc5.weight.bias = self.ALPHA_L * self.dec_fc5_delta_bias + self.ALPHA_G * (self.dec_fc5.bias.grad if self.dec_fc5.bias.grad is not None else 0.)
		self.dec_bn5.weight.grad = self.ALPHA_L * self.dec_bn5_delta_w + self.ALPHA_G * (self.dec_bn5.weight.grad if self.dec_bn5.weight.grad is not None else 0.)
		self.dec_bn5.weight.bias = self.ALPHA_L * self.dec_bn5_delta_bias + self.ALPHA_G * (self.dec_bn5.bias.grad if self.dec_bn5.bias.grad is not None else 0.)
		self.dec_conv5.weight.grad = self.ALPHA_L * self.dec_conv5_delta_w + self.ALPHA_G * (self.dec_conv5.weight.grad if self.dec_conv5.weight.grad is not None else 0.)
		self.dec_conv5.weight.bias = self.ALPHA_L * self.dec_conv5_delta_bias + self.ALPHA_G * (self.dec_conv5.bias.grad if self.dec_conv5.bias.grad is not None else 0.)
		
		self.conv4.weight.grad = self.ALPHA_L * self.conv4_delta_w + self.ALPHA_G * (self.conv4.weight.grad if self.conv4.weight.grad is not None else 0.)
		self.conv4.weight.bias = self.ALPHA_L * self.conv4_delta_bias + self.ALPHA_G * (self.conv4.bias.grad if self.conv4.bias.grad is not None else 0.)
		self.bn4.weight.grad = self.ALPHA_L * self.bn4_delta_w + self.ALPHA_G * (self.bn4.weight.grad if self.bn4.weight.grad is not None else 0.)
		self.bn4.weight.bias = self.ALPHA_L * self.bn4_delta_bias + self.ALPHA_G * (self.bn4.bias.grad if self.bn4.bias.grad is not None else 0.)
		self.fc_mu4.weight.grad = self.ALPHA_L * self.fc_mu4_delta_w + self.ALPHA_G * (self.fc_mu4.weight.grad if self.fc_mu4.weight.grad is not None else 0.)
		self.fc_mu4.weight.bias = self.ALPHA_L * self.fc_mu4_delta_bias + self.ALPHA_G * (self.fc_mu4.bias.grad if self.fc_mu4.bias.grad is not None else 0.)
		self.fc_var4.weight.grad = self.ALPHA_L * self.fc_var4_delta_w + self.ALPHA_G * (self.fc_var4.weight.grad if self.fc_var4.weight.grad is not None else 0.)
		self.fc_var4.weight.bias = self.ALPHA_L * self.fc_var4_delta_bias + self.ALPHA_G * (self.fc_var4.bias.grad if self.fc_var4.bias.grad is not None else 0.)
		self.dec_fc6.weight.grad = self.ALPHA_L * self.dec_fc6_delta_w + self.ALPHA_G * (self.dec_fc6.weight.grad if self.dec_fc6.weight.grad is not None else 0.)
		self.dec_fc6.weight.bias = self.ALPHA_L * self.dec_fc6_delta_bias + self.ALPHA_G * (self.dec_fc6.bias.grad if self.dec_fc6.bias.grad is not None else 0.)
		self.dec_bn6.weight.grad = self.ALPHA_L * self.dec_bn6_delta_w + self.ALPHA_G * (self.dec_bn6.weight.grad if self.dec_bn6.weight.grad is not None else 0.)
		self.dec_bn6.weight.bias = self.ALPHA_L * self.dec_bn6_delta_bias + self.ALPHA_G * (self.dec_bn6.bias.grad if self.dec_bn6.bias.grad is not None else 0.)
		self.dec_conv6.weight.grad = self.ALPHA_L * self.dec_conv6_delta_w + self.ALPHA_G * (self.dec_conv6.weight.grad if self.dec_conv6.weight.grad is not None else 0.)
		self.dec_conv6.weight.bias = self.ALPHA_L * self.dec_conv6_delta_bias + self.ALPHA_G * (self.dec_conv6.bias.grad if self.dec_conv6.bias.grad is not None else 0.)
		
		self.conv3.weight.grad = self.ALPHA_L * self.conv3_delta_w + self.ALPHA_G * (self.conv3.weight.grad if self.conv3.weight.grad is not None else 0.)
		self.conv3.weight.bias = self.ALPHA_L * self.conv3_delta_bias + self.ALPHA_G * (self.conv3.bias.grad if self.conv3.bias.grad is not None else 0.)
		self.bn3.weight.grad = self.ALPHA_L * self.bn3_delta_w + self.ALPHA_G * (self.bn3.weight.grad if self.bn3.weight.grad is not None else 0.)
		self.bn3.weight.bias = self.ALPHA_L * self.bn3_delta_bias + self.ALPHA_G * (self.bn3.bias.grad if self.bn3.bias.grad is not None else 0.)
		self.fc_mu3.weight.grad = self.ALPHA_L * self.fc_mu3_delta_w + self.ALPHA_G * (self.fc_mu3.weight.grad if self.fc_mu3.weight.grad is not None else 0.)
		self.fc_mu3.weight.bias = self.ALPHA_L * self.fc_mu3_delta_bias + self.ALPHA_G * (self.fc_mu3.bias.grad if self.fc_mu3.bias.grad is not None else 0.)
		self.fc_var3.weight.grad = self.ALPHA_L * self.fc_var3_delta_w + self.ALPHA_G * (self.fc_var3.weight.grad if self.fc_var3.weight.grad is not None else 0.)
		self.fc_var3.weight.bias = self.ALPHA_L * self.fc_var3_delta_bias + self.ALPHA_G * (self.fc_var3.bias.grad if self.fc_var3.bias.grad is not None else 0.)
		self.dec_fc7.weight.grad = self.ALPHA_L * self.dec_fc7_delta_w + self.ALPHA_G * (self.dec_fc7.weight.grad if self.dec_fc7.weight.grad is not None else 0.)
		self.dec_fc7.weight.bias = self.ALPHA_L * self.dec_fc7_delta_bias + self.ALPHA_G * (self.dec_fc7.bias.grad if self.dec_fc7.bias.grad is not None else 0.)
		self.dec_bn7.weight.grad = self.ALPHA_L * self.dec_bn7_delta_w + self.ALPHA_G * (self.dec_bn7.weight.grad if self.dec_bn7.weight.grad is not None else 0.)
		self.dec_bn7.weight.bias = self.ALPHA_L * self.dec_bn7_delta_bias + self.ALPHA_G * (self.dec_bn7.bias.grad if self.dec_bn7.bias.grad is not None else 0.)
		self.dec_conv7.weight.grad = self.ALPHA_L * self.dec_conv7_delta_w + self.ALPHA_G * (self.dec_conv7.weight.grad if self.dec_conv7.weight.grad is not None else 0.)
		self.dec_conv7.weight.bias = self.ALPHA_L * self.dec_conv7_delta_bias + self.ALPHA_G * (self.dec_conv7.bias.grad if self.dec_conv7.bias.grad is not None else 0.)
		
		self.conv2.weight.grad = self.ALPHA_L * self.conv2_delta_w + self.ALPHA_G * (self.conv2.weight.grad if self.conv2.weight.grad is not None else 0.)
		self.conv2.weight.bias = self.ALPHA_L * self.conv2_delta_bias + self.ALPHA_G * (self.conv2.bias.grad if self.conv2.bias.grad is not None else 0.)
		self.bn2.weight.grad = self.ALPHA_L * self.bn2_delta_w + self.ALPHA_G * (self.bn2.weight.grad if self.bn2.weight.grad is not None else 0.)
		self.bn2.weight.bias = self.ALPHA_L * self.bn2_delta_bias + self.ALPHA_G * (self.bn2.bias.grad if self.bn2.bias.grad is not None else 0.)
		self.fc_mu2.weight.grad = self.ALPHA_L * self.fc_mu2_delta_w + self.ALPHA_G * (self.fc_mu2.weight.grad if self.fc_mu2.weight.grad is not None else 0.)
		self.fc_mu2.weight.bias = self.ALPHA_L * self.fc_mu2_delta_bias + self.ALPHA_G * (self.fc_mu2.bias.grad if self.fc_mu2.bias.grad is not None else 0.)
		self.fc_var2.weight.grad = self.ALPHA_L * self.fc_var2_delta_w + self.ALPHA_G * (self.fc_var2.weight.grad if self.fc_var2.weight.grad is not None else 0.)
		self.fc_var2.weight.bias = self.ALPHA_L * self.fc_var2_delta_bias + self.ALPHA_G * (self.fc_var2.bias.grad if self.fc_var2.bias.grad is not None else 0.)
		self.dec_fc8.weight.grad = self.ALPHA_L * self.dec_fc8_delta_w + self.ALPHA_G * (self.dec_fc8.weight.grad if self.dec_fc8.weight.grad is not None else 0.)
		self.dec_fc8.weight.bias = self.ALPHA_L * self.dec_fc8_delta_bias + self.ALPHA_G * (self.dec_fc8.bias.grad if self.dec_fc8.bias.grad is not None else 0.)
		self.dec_bn8.weight.grad = self.ALPHA_L * self.dec_bn8_delta_w + self.ALPHA_G * (self.dec_bn8.weight.grad if self.dec_bn8.weight.grad is not None else 0.)
		self.dec_bn8.weight.bias = self.ALPHA_L * self.dec_bn8_delta_bias + self.ALPHA_G * (self.dec_bn8.bias.grad if self.dec_bn8.bias.grad is not None else 0.)
		self.dec_conv8.weight.grad = self.ALPHA_L * self.dec_conv8_delta_w + self.ALPHA_G * (self.dec_conv8.weight.grad if self.dec_conv8.weight.grad is not None else 0.)
		self.dec_conv8.weight.bias = self.ALPHA_L * self.dec_conv8_delta_bias + self.ALPHA_G * (self.dec_conv8.bias.grad if self.dec_conv8.bias.grad is not None else 0.)
		
		self.conv1.weight.grad = self.ALPHA_L * self.conv1_delta_w + self.ALPHA_G * (self.conv1.weight.grad if self.conv1.weight.grad is not None else 0.)
		self.conv1.weight.bias = self.ALPHA_L * self.conv1_delta_bias + self.ALPHA_G * (self.conv1.bias.grad if self.conv1.bias.grad is not None else 0.)
		self.bn1.weight.grad = self.ALPHA_L * self.bn1_delta_w + self.ALPHA_G * (self.bn1.weight.grad if self.bn1.weight.grad is not None else 0.)
		self.bn1.weight.bias = self.ALPHA_L * self.bn1_delta_bias + self.ALPHA_G * (self.bn1.bias.grad if self.bn1.bias.grad is not None else 0.)
		self.fc_mu1.weight.grad = self.ALPHA_L * self.fc_mu1_delta_w + self.ALPHA_G * (self.fc_mu1.weight.grad if self.fc_mu1.weight.grad is not None else 0.)
		self.fc_mu1.weight.bias = self.ALPHA_L * self.fc_mu1_delta_bias + self.ALPHA_G * (self.fc_mu1.bias.grad if self.fc_mu1.bias.grad is not None else 0.)
		self.fc_var1.weight.grad = self.ALPHA_L * self.fc_var1_delta_w + self.ALPHA_G * (self.fc_var1.weight.grad if self.fc_var1.weight.grad is not None else 0.)
		self.fc_var1.weight.bias = self.ALPHA_L * self.fc_var1_delta_bias + self.ALPHA_G * (self.fc_var1.bias.grad if self.fc_var1.bias.grad is not None else 0.)
		self.dec_fc9.weight.grad = self.ALPHA_L * self.dec_fc9_delta_w + self.ALPHA_G * (self.dec_fc9.weight.grad if self.dec_fc9.weight.grad is not None else 0.)
		self.dec_fc9.weight.bias = self.ALPHA_L * self.dec_fc9_delta_bias + self.ALPHA_G * (self.dec_fc9.bias.grad if self.dec_fc9.bias.grad is not None else 0.)
		self.dec_bn9.weight.grad = self.ALPHA_L * self.dec_bn9_delta_w + self.ALPHA_G * (self.dec_bn9.weight.grad if self.dec_bn9.weight.grad is not None else 0.)
		self.dec_bn9.weight.bias = self.ALPHA_L * self.dec_bn9_delta_bias + self.ALPHA_G * (self.dec_bn9.bias.grad if self.dec_bn9.bias.grad is not None else 0.)
		self.dec_conv9.weight.grad = self.ALPHA_L * self.dec_conv9_delta_w + self.ALPHA_G * (self.dec_conv9.weight.grad if self.dec_conv9.weight.grad is not None else 0.)
		self.dec_conv9.weight.bias = self.ALPHA_L * self.dec_conv9_delta_bias + self.ALPHA_G * (self.dec_conv9.bias.grad if self.dec_conv9.bias.grad is not None else 0.)
		