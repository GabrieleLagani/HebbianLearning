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
	FC1 = 'fc1'
	BN1 = 'bn1'
	FC2 = 'fc2'
	CLF_OUTPUT = 'clf_output' # Name of the classification output providing the class scores
	
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
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.fc1 = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_channels=4096,
			kernel_size=(self.get_input_shape()[1], self.get_input_shape()[2]) if len(self.get_input_shape()) >= 3 else 1,
			lrn_sim=self.lrn_sim,
			lrn_act=self.lrn_act,
			lrn_cmp=True,
			lrn_t=True,
			out_sim=self.out_sim,
			out_act=self.out_act,
			competitive=H.Competitive(out_size=HF.get_factors(self.NUM_HIDDEN), competitive_act=self.competitive_act, k=self.K),
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
		)  # input_shape-shaped input, 64x64=self.NUM_HIDDEN output channels
		self.bn1 = nn.BatchNorm2d(self.NUM_HIDDEN)  # Batch Norm layer
		
		self.fc2 = H.HebbianConv2d(
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
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Hidden Layer: FC + Batch Norm
		fc1_out = self.fc1(x if len(self.get_input_shape()) >= 3 else x.view(x.size(0), x.size(1), 1, 1))
		bn1_out = HF.modified_bn(self.bn1, fc1_out)
		
		# Output Layer, outputs are the class scores
		fc2_out = self.fc2(bn1_out).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC1] = fc1_out
		out[self.BN1] = bn1_out
		out[self.FC2] = fc2_out
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: fc2_out}
		return out
	
	def set_teacher_signal(self, y):
		if isinstance(y, dict): y = y[P.KEY_LABEL_TARGETS]
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		
		self.fc2.set_teacher_signal(y)
		if y is None:
			self.fc1.set_teacher_signal(y)
		elif self.DEEP_TEACHER_SIGNAL:
			# Extend teacher signal for deep layers
			l1_knl_per_class = self.NUM_HIDDEN // self.NUM_CLASSES
			self.fc1.set_teacher_signal(
				torch.cat((
					torch.ones(y.size(0), self.fc1.weight.size(0) - l1_knl_per_class * self.NUM_CLASSES, device=y.device),
					y.view(y.size(0), y.size(1), 1).repeat(1, 1, l1_knl_per_class).view(y.size(0), -1),
				), dim=1)
			)

	def local_updates(self):
		self.fc1.local_update()
		self.fc2.local_update()

