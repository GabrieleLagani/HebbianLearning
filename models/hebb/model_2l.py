import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
import params as PP
from neurolab import utils
from neurolab.model import Model
import hebb as H
from hebb import functional as HF

class Net(Model):
	# Layer names
	CONV1 = 'conv1'
	POOL1 = 'pool1'
	BN1 = 'bn1'
	CONV_OUTPUT = BN1  # Symbolic name for the last convolutional layer providing extracted features
	FC2 = 'fc2'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.DEEP_TEACHER_SIGNAL = config.CONFIG_OPTIONS.get(P.KEY_DEEP_TEACHER_SIGNAL, False)
		self.LRN_SIM = HF.kernel_mult2d
		self.LRN_ACT = F.relu
		self.OUT_SIM = HF.kernel_mult2d
		self.OUT_ACT = F.relu
		self.COMPETITIVE_ACT = None
		self.K = 0
		self.ACT_COMPLEMENT_INIT = None
		self.ACT_COMPLEMENT_RATIO = 0.
		self.ACT_COMPLEMENT_ADAPT = None
		self.ACT_COMPLEMENT_GRP = False
		self.GATING = H.HebbianConv2d.GATE_HEBB
		self.UPD_RULE = H.HebbianConv2d.UPD_RECONSTR
		self.RECONSTR = H.HebbianConv2d.REC_LIN_CMB
		self.RED = H.HebbianConv2d.RED_AVG
		self.VAR_ADAPTIVE = False
		self.LAMB = config.CONFIG_OPTIONS.get(PP.KEY_ACT_LAMB, 1.)
		self.LOC_LRN_RULE = config.CONFIG_OPTIONS.get(P.KEY_LOCAL_LRN_RULE, 'hpca')
		if self.LOC_LRN_RULE in ['hpcat', 'hpcat_ada']:
			self.LRN_ACT = HF.tanh
			self.OUT_ACT = HF.tanh
			if self.LOC_LRN_RULE == 'hpcat_ada': self.VAR_ADAPTIVE = True
		if self.LOC_LRN_RULE == 'hwta':
			self.LRN_SIM = HF.raised_cos2d_pow(2)
			self.LRN_ACT = HF.identity
			self.OUT_SIM = HF.vector_proj2d
			self.OUT_ACT = F.relu
			self.COMPETITIVE_ACT = config.CONFIG_OPTIONS.get(PP.KEY_WTA_COMPETITIVE_ACT, None)
			if self.COMPETITIVE_ACT is not None: self.COMPETITIVE_ACT = utils.retrieve(self.COMPETITIVE_ACT)
			self.K = config.CONFIG_OPTIONS.get(PP.KEY_WTA_K, 1)
			self.GATING = H.HebbianConv2d.GATE_BASE
			self.RECONSTR = H.HebbianConv2d.REC_QNT_SGN
			self.RED = H.HebbianConv2d.RED_W_AVG
		if self.LOC_LRN_RULE in ['ica', 'hica', 'ica_nrm', 'hica_nrm']:
			self.LRN_ACT = HF.tanh
			self.OUT_ACT = HF.tanh
			self.ACT_COMPLEMENT_INIT = config.CONFIG_OPTIONS.get(PP.KEY_ICA_ACT_COMPLEMENT_INIT, None)
			self.ACT_COMPLEMENT_RATIO = config.CONFIG_OPTIONS.get(PP.KEY_ICA_ACT_COMPLEMENT_RATIO, 0.)
			self.ACT_COMPLEMENT_ADAPT = config.CONFIG_OPTIONS.get(PP.KEY_ICA_ACT_COMPLEMENT_ADAPT, None)
			self.ACT_COMPLEMENT_GRP = config.CONFIG_OPTIONS.get(PP.KEY_ICA_ACT_COMPLEMENT_GRP, False)
			self.GATING = H.HebbianConv2d.GATE_BASE
			self.UPD_RULE = H.HebbianConv2d.UPD_ICA
			if self.LOC_LRN_RULE == 'hica': self.UPD_RULE = H.HebbianConv2d.UPD_HICA
			if self.LOC_LRN_RULE == 'ica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_ICA_NRM
			if self.LOC_LRN_RULE == 'hica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_HICA_NRM
			if self.LOC_LRN_RULE in ['ica_nrm', 'hica_nrm']: self.VAR_ADAPTIVE = True
		self.ALPHA = config.CONFIG_OPTIONS.get(P.KEY_ALPHA, 1.)
		
		# Here we define the layers of our network
		
		# First convolutional layer
		self.conv1 = H.HebbianConv2d(
			in_channels=3,
			out_channels=96,
			kernel_size=5,
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(8, 12), competitive_act=self.COMPETITIVE_ACT, k=self.K),
			out_sim=self.OUT_SIM,
			out_act=self.OUT_ACT,
			out_cmp=None,
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			lamb=self.LAMB,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha=self.ALPHA,
		) # 3 x channels, 8x12=96 output channels, 5x5 convolutions
		self.bn1 = nn.BatchNorm2d(96)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT])
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc2 = H.HebbianConv2d(
			in_channels=self.CONV_OUTPUT_SHAPE[0],
			out_channels=self.NUM_CLASSES,
			kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]),
			lrn_sim=HF.raised_cos2d_pow(2),
			lrn_act=HF.identity,
			lrn_cmp=H.Competitive(),
			out_sim=HF.vector_proj2d,
			out_act=HF.identity,
			out_cmp=None,
			gating=H.HebbianConv2d.GATE_BASE,
			upd_rule=H.HebbianConv2d.UPD_RECONSTR,
			reconstruction=H.HebbianConv2d.REC_QNT_SGN,
			reduction=H.HebbianConv2d.RED_W_AVG,
			alpha=self.ALPHA,
		) # conv_output_shape-shaped x, 10-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 1: Convolutional + 2x2 Max Pooling + Batch Norm
		conv1_out = self.conv1(x)
		pool1_out = F.max_pool2d(conv1_out, 2)
		bn1_out = self.bn1(pool1_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV1: conv1_out,
			self.POOL1: pool1_out,
			self.BN1: bn1_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Linear FC layer, outputs are the class scores
		fc2_out = self.fc2(out[self.CONV_OUTPUT]).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC2] = fc2_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
	
	def set_teacher_signal(self, y):
		if isinstance(y, dict): y = y[P.KEY_LABEL_TARGETS]
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		self.fc2.set_teacher_signal(y)
	
	def local_updates(self):
		self.conv1.local_update()
		self.fc2.local_update()
	
	