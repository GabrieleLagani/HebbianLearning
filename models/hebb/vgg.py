import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab.model import SimpleWrapper
import hebb as H
from hebb import functional as HF
from .base import HebbFactory
import utils
import params as PP


class Net(SimpleWrapper):
	def wrapped_init(self, config, input_shape=None):
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(P.KEY_NUM_HIDDEN, 4096)
		self.DROPOUT_P = config.CONFIG_OPTIONS.get(P.KEY_DROPOUT_P, 0.5)
		self.VGG_MODEL = config.CONFIG_OPTIONS.get(PP.KEY_VGG_MODEL, 'VGG11')
		
		return Model(input_shape=input_shape, num_classes=self.NUM_CLASSES, num_hidden=self.NUM_HIDDEN, dropout_p=self.DROPOUT_P, vgg_model=self.VGG_MODEL, hebb_param_dict=config.CONFIG_OPTIONS)

	def set_teacher_signal(self, y):
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		super().set_teacher_signal(y)

layer_seq = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Model(nn.Module):
	# Layer names
	CONV_OUTPUT = 'conv_output'
	CLF = 'clf'
	CLF_OUTPUT = 'clf_output'
	
	def __init__(self, input_shape=None, num_classes=1000, num_hidden=4096, dropout_p=0., vgg_model='VGG11', hebb_param_dict=None):
		super().__init__()
		
		self.INPUT_SHAPE = input_shape
		self.NUM_CLASSES = num_classes
		self.NUM_HIDDEN = num_hidden
		self.DROPOUT_P = dropout_p
		self.VGG_MODEL = vgg_model
		self.HEBB_PARAM_DICT = hebb_param_dict
		self.hfactory = HebbFactory(hebb_param_dict)
		
		# Here we define the layers of our network
		
		layers = []
		in_channels = 3
		for i, l in enumerate(layer_seq[self.VGG_MODEL]):
			if l == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)] # 2x2 Max Pooling layer
			else:
				# Deep convolutional layer: # in_channels input channels, l output channels, 3x3 convolutions
				layers += [
					nn.ZeroPad2d(padding=1), # Padding before convolution
					self.hfactory.create_hebb_layer(in_channels=in_channels, out_channels=l, kernel_size=3), # Convolutional layer
					nn.BatchNorm2d(l), # Batch Norm layer
				]
				in_channels = l
		
		self.features = nn.Sequential(*layers)
		
		self.CONV_OUTPUT_SHAPE = None
		self.CONV_OUTPUT_SHAPE = utils.get_output_fmap_shape(self, input_shape)[self.CONV_OUTPUT]
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.classifier = nn.Sequential(
			# Deep FC layer: self.CONV_OUTPUT_SHAPE-shaped input, self.NUM_HIDDEN output channels
			self.hfactory.create_hebb_layer(in_channels=self.CONV_OUTPUT_SHAPE[0], out_channels=self.NUM_HIDDEN, kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2])),
			HF.ModifiedBN(nn.BatchNorm2d(self.NUM_HIDDEN)),  # Batch Norm layer
			
			# Final FC layer: self.NUM_HIDDEN-dimensional input, self.NUM_CLASSES-dimensional output (one per class)
			self.hfactory.create_hebb_layer(final=True, in_channels=self.NUM_HIDDEN, out_channels=self.NUM_CLASSES, kernel_size=1, teacher_distrib=1),
		)
	
	def get_conv_output(self, x):
		return {self.CONV_OUTPUT: self.features(x)}
	
	def forward(self, x):
		out = self.get_conv_output(x)
		
		if self.CONV_OUTPUT_SHAPE is None: return out
		
		class_scores = self.classifier(out[self.CONV_OUTPUT]).view(-1, self.NUM_CLASSES)
		
		out[self.CLF] = class_scores
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: class_scores}
		return out

