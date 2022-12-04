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
		self.NUM_HIDDEN = config.CONFIG_OPTIONS.get(PP.KEY_NUM_HIDDEN, 4096)
		self.DEEP_TEACHER_SIGNAL = config.CONFIG_OPTIONS.get(P.KEY_DEEP_TEACHER_SIGNAL, False)
		LRN_SIM = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM, None)
		LRN_ACT = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT, None)
		OUT_SIM = config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM, None)
		OUT_ACT = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT, None)
		self.lrn_sim = utils.retrieve(LRN_SIM) if LRN_SIM is not None else HF.kernel_mult2d
		self.lrn_act = utils.retrieve(LRN_ACT) if LRN_ACT is not None else F.relu
		self.out_sim = utils.retrieve(OUT_SIM) if OUT_SIM is not None else HF.kernel_mult2d
		self.out_act = utils.retrieve(OUT_ACT) if OUT_ACT is not None else F.relu
		self.competitive_act = config.CONFIG_OPTIONS.get(PP.KEY_COMPETITIVE_ACT, None)
		if self.competitive_act is not None: self.competitive_act = utils.retrieve(self.competitive_act)
		self.K = config.CONFIG_OPTIONS.get(PP.KEY_COMPETITIVE_K, 1)
		self.LRN_SIM_B = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_B, 0.)
		self.LRN_SIM_S = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_S, 1.)
		self.LRN_SIM_P = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_P, 1.)
		self.LRN_SIM_EXP = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_EXP, None)
		self.LRN_ACT_SCALE_IN = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT_SCALE_IN, 1)
		self.LRN_ACT_SCALE_OUT = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT_SCALE_OUT, 1)
		self.LRN_ACT_OFFSET_IN = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT_OFFSET_IN, 0)
		self.LRN_ACT_OFFSET_OUT = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT_OFFSET_OUT, 0)
		self.LRN_ACT_P = config.CONFIG_OPTIONS.get(PP.KEY_LRN_ACT_P, 1)
		self.OUT_SIM_B = config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM_B, 0.)
		self.OUT_SIM_S = config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM_S, 1.)
		self.OUT_SIM_P = config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM_P, 1.)
		self.OUT_SIM_EXP = config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM_EXP, None)
		self.OUT_ACT_SCALE_IN = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT_SCALE_IN, 1)
		self.OUT_ACT_SCALE_OUT = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT_SCALE_OUT, 1)
		self.OUT_ACT_OFFSET_IN = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT_OFFSET_IN, 0)
		self.OUT_ACT_OFFSET_OUT = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT_OFFSET_OUT, 0)
		self.OUT_ACT_P = config.CONFIG_OPTIONS.get(PP.KEY_OUT_ACT_P, 1)
		self.ACT_COMPLEMENT_INIT = None
		self.ACT_COMPLEMENT_RATIO = 0.
		self.ACT_COMPLEMENT_ADAPT = None
		self.ACT_COMPLEMENT_GRP = False
		self.GATING = H.HebbianConv2d.GATE_HEBB
		self.UPD_RULE = H.HebbianConv2d.UPD_RECONSTR
		self.RECONSTR = H.HebbianConv2d.REC_LIN_CMB
		self.RED = H.HebbianConv2d.RED_AVG
		self.VAR_ADAPTIVE = False
		self.LOC_LRN_RULE = config.CONFIG_OPTIONS.get(P.KEY_LOCAL_LRN_RULE, 'hpca')
		if self.LOC_LRN_RULE in ['hpcat', 'hpcat_ada']:
			if LRN_ACT is None: self.lrn_act = HF.tanh
			if OUT_ACT is None: self.out_act = HF.tanh
			if self.LOC_LRN_RULE == 'hpcat_ada': self.VAR_ADAPTIVE = True
		if self.LOC_LRN_RULE == 'hwta':
			if LRN_SIM is None:
				self.lrn_sim = HF.raised_cos_sim2d
				self.LRN_SIM_P = config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_P, 2.) # NB: In hwta the default lrn_sim is squared raised cosine
			if LRN_ACT is None: self.lrn_act = HF.identity
			if OUT_SIM is None: self.out_sim = HF.vector_proj2d
			if OUT_ACT is None: self.out_act = F.relu
			self.GATING = H.HebbianConv2d.GATE_BASE
			self.RECONSTR = H.HebbianConv2d.REC_QNT_SGN
			self.RED = H.HebbianConv2d.RED_W_AVG
		if self.LOC_LRN_RULE in ['ica', 'hica', 'ica_nrm', 'hica_nrm']:
			if LRN_ACT is None: self.lrn_act = HF.tanh
			if OUT_ACT is None: self.out_act = HF.tanh
			self.ACT_COMPLEMENT_INIT = config.CONFIG_OPTIONS.get(PP.KEY_ACT_COMPLEMENT_INIT, None)
			self.ACT_COMPLEMENT_RATIO = config.CONFIG_OPTIONS.get(PP.KEY_ACT_COMPLEMENT_RATIO, 0.)
			self.ACT_COMPLEMENT_ADAPT = config.CONFIG_OPTIONS.get(PP.KEY_ACT_COMPLEMENT_ADAPT, None)
			self.ACT_COMPLEMENT_GRP = config.CONFIG_OPTIONS.get(PP.KEY_ACT_COMPLEMENT_GRP, False)
			self.UPD_RULE = H.HebbianConv2d.UPD_ICA
			if self.LOC_LRN_RULE == 'hica': self.UPD_RULE = H.HebbianConv2d.UPD_HICA
			if self.LOC_LRN_RULE == 'ica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_ICA_NRM
			if self.LOC_LRN_RULE == 'hica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_HICA_NRM
			if self.LOC_LRN_RULE in ['ica_nrm', 'hica_nrm']: self.VAR_ADAPTIVE = True
			self.GATING = H.HebbianConv2d.GATE_BASE
		if self.LRN_SIM_EXP is not None: self.lrn_sim = HF.get_exp_sim(HF.get_affine_sim(self.lrn_sim, p=self.LRN_SIM_EXP), HF.get_pow_nc(utils.retrieve(config.CONFIG_OPTIONS.get(PP.KEY_LRN_SIM_NC, None)), self.LRN_SIM_EXP))
		self.lrn_sim = HF.get_affine_sim(self.lrn_sim, self.LRN_SIM_B, self.LRN_SIM_S, self.LRN_SIM_P)
		self.lrn_act = HF.get_affine_act(self.lrn_act, self.LRN_ACT_SCALE_IN, self.LRN_ACT_SCALE_OUT, self.LRN_ACT_OFFSET_IN, self.LRN_ACT_OFFSET_OUT, self.LRN_ACT_P)
		if self.OUT_SIM_EXP is not None: self.out_sim = HF.get_exp_sim(HF.get_affine_sim(self.out_sim, p=self.OUT_SIM_EXP), HF.get_pow_nc(utils.retrieve(config.CONFIG_OPTIONS.get(PP.KEY_OUT_SIM_NC, None)), self.OUT_SIM_EXP))
		self.out_sim = HF.get_affine_sim(self.out_sim, self.OUT_SIM_B, self.OUT_SIM_S, self.OUT_SIM_P)
		self.out_act = HF.get_affine_act(self.out_act, self.OUT_ACT_SCALE_IN, self.OUT_ACT_SCALE_OUT, self.OUT_ACT_OFFSET_IN, self.OUT_ACT_OFFSET_OUT, self.OUT_ACT_P)
		self.ALPHA_L = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_L, 1.)
		self.ALPHA_G = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_G, 0.)
		
		# Here we define the layers of our network
		
		# Second convolutional layer
		self.conv2 = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_channels=128,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(8, 16), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 96 input channels, 8x16=128 output channels, 3x3 convolutions
		self.bn2 = nn.BatchNorm2d(128)  # Batch Norm layer
		
		# Third convolutional layer
		self.conv3 = H.HebbianConv2d(
			in_channels=128,
			out_channels=192,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(12, 16), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 128 input channels, 12x16=192 output channels, 3x3 convolutions
		self.bn3 = nn.BatchNorm2d(192)  # Batch Norm layer
		
		# Fourth convolutional layer
		self.conv4 = H.HebbianConv2d(
			in_channels=192,
			out_channels=192,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(12, 16), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 192 input channels, 12x16=192 output channels, 3x3 convolutions
		self.bn4 = nn.BatchNorm2d(192)  # Batch Norm layer
		
		# Fifth convolutional layer
		self.conv5 = H.HebbianConv2d(
			in_channels=192,
			out_channels=256,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(16, 16), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 192 input channels, 16x16=256 output channels, 3x3 convolutions
		self.bn5 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		# Sixth convolutional layer
		self.conv6 = H.HebbianConv2d(
			in_channels=256,
			out_channels=256,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(16, 16), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 256 input channels, 16x16=256 output channels, 3x3 convolutions
		self.bn6 = nn.BatchNorm2d(256)  # Batch Norm layer
		
		# Seventh convolutional layer
		self.conv7 = H.HebbianConv2d(
			in_channels=256,
			out_channels=384,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(16, 24), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 256 input channels, 16x24=384 output channels, 3x3 convolutions
		self.bn7 = nn.BatchNorm2d(384)  # Batch Norm layer
		
		# Eighth convolutional layer
		self.conv8 = H.HebbianConv2d(
			in_channels=384,
			out_channels=512,
			kernel_size=3,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=(16, 32), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # 384 input channels, 16x32=512 output channels, 3x3 convolutions
		self.bn8 = nn.BatchNorm2d(512)  # Batch Norm layer
		
		self.CONV_OUTPUT_SHAPE = utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT])
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc9 = H.HebbianConv2d(
			in_channels=self.CONV_OUTPUT_SHAPE[0],
			out_channels=self.NUM_HIDDEN,
			kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]),
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=utils.get_factors(self.NUM_HIDDEN), competitive_act=self.competitive_act, k=self.K),
			act_complement_init=self.ACT_COMPLEMENT_INIT,
			act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
			act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
			act_complement_grp=self.ACT_COMPLEMENT_GRP,
			var_adaptive=self.VAR_ADAPTIVE,
			gating=self.GATING,
			upd_rule=self.UPD_RULE,
			reconstruction=self.RECONSTR,
			reduction=self.RED,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G,
		)  # conv_output_shape-shaped input, 64x64=self.NUM_HIDDEN output channels
		self.bn9 = nn.BatchNorm2d(self.NUM_HIDDEN)  # Batch Norm layer
		
		self.fc10 = H.HebbianConv2d(
			in_channels=self.NUM_HIDDEN,
			out_channels=self.NUM_CLASSES,
			kernel_size=1,
			lrn_sim=HF.get_affine_sim(HF.raised_cos_sim2d, p=2),
			lrn_act=HF.identity,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=HF.vector_proj2d if self.ALPHA_G == 0. else HF.kernel_mult2d,
			out_act=HF.identity,
			competitive=H.Competitive(),
			gating=H.HebbianConv2d.GATE_BASE,
			upd_rule=H.HebbianConv2d.UPD_RECONSTR if self.ALPHA_G == 0. else None,
			reconstruction=H.HebbianConv2d.REC_QNT_SGN,
			reduction=H.HebbianConv2d.RED_W_AVG,
			alpha_l=self.ALPHA_L,
			alpha_g=self.ALPHA_G if self.ALPHA_G == 0. else 1.,
		)  # self.NUM_HIDDEN-dimensional input, NUM_CLASSES-dimensional output (one per class)
	
	def get_conv_output(self, x):
		# Layer 2: Convolutional + Batch Norm
		conv2_out = self.conv2(x)
		bn2_out = self.bn2(conv2_out)
		
		# Layer 3: Convolutional + 2x2 Max Pooling + Batch Norm
		conv3_out = self.conv3(bn2_out)
		pool3_out = F.max_pool2d(conv3_out, 2)
		bn3_out = self.bn3(pool3_out)
		
		# Layer 4: Convolutional + Batch Norm
		conv4_out = self.conv4(bn3_out)
		bn4_out = self.bn4(conv4_out)
		
		# Layer 5: Convolutional + 2x2 Max Pooling + Batch Norm
		conv5_out = self.conv5(bn4_out)
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
			self.CONV2: conv2_out,
			self.BN2: bn2_out,
			self.CONV3: conv3_out,
			self.POOL3: pool3_out,
			self.BN3: bn3_out,
			self.CONV4: conv4_out,
			self.BN4: bn4_out,
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
			self.conv2.set_teacher_signal(y)
			self.conv3.set_teacher_signal(y)
			self.conv4.set_teacher_signal(y)
			self.conv5.set_teacher_signal(y)
			self.conv6.set_teacher_signal(y)
			self.conv7.set_teacher_signal(y)
			self.conv8.set_teacher_signal(y)
			self.fc9.set_teacher_signal(y)
		elif self.DEEP_TEACHER_SIGNAL:
			# Extend teacher signal for deep layers
			l2_knl_per_class = 80 // self.NUM_CLASSES
			l3_knl_per_class = 160 // self.NUM_CLASSES
			l4_knl_per_class = 160 // self.NUM_CLASSES
			l5_knl_per_class = 240 // self.NUM_CLASSES
			l6_knl_per_class = 240 // self.NUM_CLASSES
			l7_knl_per_class = 360 // self.NUM_CLASSES
			l8_knl_per_class = 500 // self.NUM_CLASSES
			l9_knl_per_class = self.NUM_HIDDEN // self.NUM_CLASSES
			if self.NUM_CLASSES <= 20:
				self.conv2.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv2.weight.size(0) - l2_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l2_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
				self.conv3.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv3.weight.size(0) - l3_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l3_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
				self.conv4.set_teacher_signal(
					torch.cat((
						torch.ones(y.size(0), self.conv4.weight.size(0) - l4_knl_per_class * self.NUM_CLASSES, device=y.device),
						y.view(y.size(0), y.size(1), 1).repeat(1, 1, l4_knl_per_class).view(y.size(0), -1),
					), dim=1)
				)
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
		self.conv2.local_update()
		self.conv3.local_update()
		self.conv4.local_update()
		self.conv5.local_update()
		self.conv6.local_update()
		self.conv7.local_update()
		self.conv8.local_update()
		self.fc9.local_update()
		self.fc10.local_update()

