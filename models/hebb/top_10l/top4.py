import torch
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
		
		# Fifth convolutional layer
		self.conv5 = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_channels=256,
			kernel_size=3,
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(16, 16), competitive_act=self.COMPETITIVE_ACT, k=self.K),
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
		)  # 192 x channels, 16x16=256 output channels, 3x3 convolutions
		self.bn5 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		# Sixth convolutional layer
		self.conv6 = H.HebbianConv2d(
			in_channels=256,
			out_channels=256,
			kernel_size=3,
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(16, 16), competitive_act=self.COMPETITIVE_ACT, k=self.K),
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
		)  # 256 x channels, 16x16=256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		# Seventh convolutional layer
		self.conv7 = H.HebbianConv2d(
			in_channels=256,
			out_channels=384,
			kernel_size=3,
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(16, 24), competitive_act=self.COMPETITIVE_ACT, k=self.K),
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
		)  # 256 x channels, 16x24=384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384)  # Batch Norm layer
		
		# Eighth convolutional layer
		self.conv8 = H.HebbianConv2d(
			in_channels=384,
			out_channels=512,
			kernel_size=3,
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(16, 32), competitive_act=self.COMPETITIVE_ACT, k=self.K),
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
		)  # 384 x channels, 16x32=512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT])
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc9 = H.HebbianConv2d(
			in_channels=self.CONV_OUTPUT_SHAPE[0],
			out_channels=4096,
			kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]),
			lrn_sim=self.LRN_SIM,
			lrn_act=self.LRN_ACT,
			lrn_cmp=H.Competitive(out_size=(64, 64), competitive_act=self.COMPETITIVE_ACT, k=self.K),
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
		)  # conv_output_shape-shaped x, 64x64=4096 output channels
		self.bn9 = nn.BatchNorm2d(4096)  # Batch Norm layer
		
		self.fc10 = H.HebbianConv2d(
			in_channels=4096,
			out_channels=self.NUM_CLASSES,
			kernel_size=1,
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
		)  # 4096-dimensional x, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 5: Convolutional + 2x2 Max Pooling + Batch Norm
		conv5_out = self.conv5(x)
		pool5_out = F.max_pool2d(conv5_out, 2)
		bn5_out = self.bn5(pool5_out)
		
		# Layer 6: Convolutional + Batch Norm
		conv6_out = self.conv6(bn5_out)
		bn6_out = self.bn6(conv6_out)
		
		# Layer 7: Convolutional + 2x2 Max Pooling + Batch Norm
		conv7_out = self.conv7(bn6_out)
		pool7_out = F.max_pool2d(conv7_out, 2)
		bn7_out = HF.modified_bn(self.bn7, pool7_out)
		
		# Layer 8: Convolutional + Batch Norm
		conv8_out = self.conv8(bn7_out)
		bn8_out = HF.modified_bn(self.bn8, conv8_out)
		
		# Build dictionary containing outputs of each layer
		conv_out = {
			self.CONV5: conv5_out,
			self.POOL5: pool5_out,
			self.BN5: bn5_out,
			self.CONV6: conv6_out,
			self.BN6: bn6_out,
			self.CONV7: conv7_out,
			self.POOL7: pool7_out,
			self.BN7: bn7_out,
			self.CONV8: conv8_out,
			self.BN8: bn8_out,
		}
		return conv_out
	
	# Here we define the flow of information through the network
	def forward(self, x):
		# Compute the output feature map from the convolutional layers
		out = self.get_conv_output(x)
		
		# Layer 9: FC + Batch Norm
		fc9_out = self.fc9(out[self.CONV_OUTPUT])
		bn9_out = HF.modified_bn(self.bn9, fc9_out)
		
		# Linear FC layer, outputs are the class scores
		fc10_out = self.fc10(bn9_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC9] = fc9_out
		out[self.BN9] = bn9_out
		out[self.FC10] = fc10_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc10_out}
		return out
	
	def set_teacher_signal(self, y):
		if isinstance(y, dict): y = y[P.KEY_LABEL_TARGETS]
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		
		self.fc10.set_teacher_signal(y)
		if y is None:
			self.conv5.set_teacher_signal(y)
			self.conv6.set_teacher_signal(y)
			self.conv7.set_teacher_signal(y)
			self.conv8.set_teacher_signal(y)
			self.fc9.set_teacher_signal(y)
		elif self.DEEP_TEACHER_SIGNAL:
			# Extend teacher signal for deep layers
			l5_knl_per_class = 240 // self.NUM_CLASSES
			l6_knl_per_class = 240 // self.NUM_CLASSES
			l7_knl_per_class = 360 // self.NUM_CLASSES
			l8_knl_per_class = 500 // self.NUM_CLASSES
			l9_knl_per_class = 4000 // self.NUM_CLASSES
			if self.NUM_CLASSES <= 20:
				self.conv5.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv5.weight.size(0) - l5_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l5_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
				self.conv6.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv6.weight.size(0) - l6_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l6_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
				self.conv7.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv7.weight.size(0) - l7_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l7_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
				self.conv8.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv8.weight.size(0) - l8_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l8_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
			self.fc9.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.fc9.weight.size(0) - l9_knl_per_class * self.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l9_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)

	def local_updates(self):
		self.conv5.local_update()
		self.conv6.local_update()
		self.conv7.local_update()
		self.conv8.local_update()
		self.fc9.local_update()
		self.fc10.local_update()

