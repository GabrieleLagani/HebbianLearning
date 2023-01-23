import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
import params as PP
from neurolab import utils
from neurolab.model import Model
import hebb as H
from hebb import functional as HF


layer_seq = {
	'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
	'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
	'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Net(Model):
	# Layer names
	CONV_OUTPUT = 'conv_output'
	CLF = 'clf'
	CLF_OUTPUT = 'clf_output'
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.VGG_MODEL = config.CONFIG_OPTIONS.get(PP.KEY_VGG_MODEL, 'VGG11')
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
		
		layers = []
		in_channels = 3
		for i, l in enumerate(layer_seq[self.VGG_MODEL]):
			if l == 'M':
				layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
			else:
				layers += [
					nn.ZeroPad2d(padding=1),
					H.HebbianConv2d(
						in_channels=in_channels,
						out_channels=l,
						kernel_size=3,
						lrn_sim=self.lrn_sim,
						lrn_act=self.lrn_act,
						lrn_cmp=True,
						lrn_t=True,
						out_sim=self.out_sim,
						out_act=self.out_act,
						competitive=H.Competitive(out_size=HF.get_factors(l), competitive_act=self.competitive_act, k=self.K),
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
					),
					nn.BatchNorm2d(l),
					nn.ReLU(inplace=True)
				]
				in_channels = l
			
		
		self.features = nn.Sequential(*layers)
		
		self.CONV_OUTPUT_SHAPE = utils.tens2shape(self.get_dummy_fmap()[self.CONV_OUTPUT])
		
		# FC Layers (convolution with kernel size equal to the entire feature map size is like a fc layer)
		
		self.classifier = nn.Sequential(
			H.HebbianConv2d(
				in_channels=self.CONV_OUTPUT_SHAPE[0],
				out_channels=self.NUM_HIDDEN,
				kernel_size=(self.CONV_OUTPUT_SHAPE[1], self.CONV_OUTPUT_SHAPE[2]),
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
			),  # conv_output_shape-shaped input, 64x64=self.NUM_HIDDEN output channels
			HF.ModifiedBN(nn.BatchNorm2d(self.NUM_HIDDEN)),  # Batch Norm layer
			
			H.HebbianConv2d(
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
		)
	
	def get_conv_output(self, x):
		return {self.CONV_OUTPUT: self.features(x)}
	
	def forward(self, x):
		out = self.get_conv_output(x)
		class_scores = self.classifier(out[self.CONV_OUTPUT]).view(-1, self.NUM_CLASSES)
		
		out[self.CLF] = class_scores
		out[self.CLF_OUTPUT] = {P.KEY_CLASS_SCORES: class_scores}
		return out
	
	def set_teacher_signal(self, y):
		if isinstance(y, dict): y = y[P.KEY_LABEL_TARGETS]
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		
		self.classifier[-1].set_teacher_signal(y)

	def local_updates(self):
		for l in self.features:
			if hasattr(l, "local_update"): l.local_update()
		for l in self.classifier:
			if hasattr(l, "local_update"): l.local_update()

