import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import params as P
from neurolab import utils
import params as PP
import hebb as H
from hebb import functional as HF


DEFAULT_HEBB_PARAM_DICT = {}

# A class for constructing neural networks based on Hebbian layers
class HebbFactory:
	def __init__(self, hebb_param_dict=None):
		self.HEBB_PARAM_DICT = hebb_param_dict if hebb_param_dict is not None else DEFAULT_HEBB_PARAM_DICT
		self.DEEP_TEACHER_SIGNAL = self.HEBB_PARAM_DICT.get(P.KEY_DEEP_TEACHER_SIGNAL, False)
		LRN_SIM = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM, None)
		LRN_ACT = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT, None)
		OUT_SIM = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM, None)
		OUT_ACT = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT, None)
		self.lrn_sim = utils.retrieve(LRN_SIM) if LRN_SIM is not None else HF.kernel_mult2d
		self.lrn_act = utils.retrieve(LRN_ACT) if LRN_ACT is not None else F.relu
		self.out_sim = utils.retrieve(OUT_SIM) if OUT_SIM is not None else HF.kernel_mult2d
		self.out_act = utils.retrieve(OUT_ACT) if OUT_ACT is not None else F.relu
		self.competitive_act = self.HEBB_PARAM_DICT.get(PP.KEY_COMPETITIVE_ACT, None)
		if self.competitive_act is not None: self.competitive_act = utils.retrieve(self.competitive_act)
		self.K = self.HEBB_PARAM_DICT.get(PP.KEY_COMPETITIVE_K, 1)
		self.LRN_SIM_B = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_B, 0.)
		self.LRN_SIM_S = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_S, 1.)
		self.LRN_SIM_P = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_P, 1.)
		self.LRN_SIM_EXP = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_EXP, None)
		self.LRN_ACT_SCALE_IN = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT_SCALE_IN, 1)
		self.LRN_ACT_SCALE_OUT = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT_SCALE_OUT, 1)
		self.LRN_ACT_OFFSET_IN = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT_OFFSET_IN, 0)
		self.LRN_ACT_OFFSET_OUT = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT_OFFSET_OUT, 0)
		self.LRN_ACT_P = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_ACT_P, 1)
		self.OUT_SIM_B = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM_B, 0.)
		self.OUT_SIM_S = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM_S, 1.)
		self.OUT_SIM_P = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM_P, 1.)
		self.OUT_SIM_EXP = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM_EXP, None)
		self.OUT_ACT_SCALE_IN = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT_SCALE_IN, 1)
		self.OUT_ACT_SCALE_OUT = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT_SCALE_OUT, 1)
		self.OUT_ACT_OFFSET_IN = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT_OFFSET_IN, 0)
		self.OUT_ACT_OFFSET_OUT = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT_OFFSET_OUT, 0)
		self.OUT_ACT_P = self.HEBB_PARAM_DICT.get(PP.KEY_OUT_ACT_P, 1)
		self.ACT_COMPLEMENT_INIT = None
		self.ACT_COMPLEMENT_RATIO = 0.
		self.ACT_COMPLEMENT_ADAPT = None
		self.ACT_COMPLEMENT_GRP = False
		self.GATING = H.HebbianConv2d.GATE_HEBB
		self.UPD_RULE = H.HebbianConv2d.UPD_RECONSTR
		self.RECONSTR = H.HebbianConv2d.REC_LIN_CMB
		self.RED = H.HebbianConv2d.RED_AVG
		self.VAR_ADAPTIVE = False
		self.LOC_LRN_RULE = self.HEBB_PARAM_DICT.get(P.KEY_LOCAL_LRN_RULE, 'hpca')
		if self.LOC_LRN_RULE in ['hpcat', 'hpcat_ada']:
			if LRN_ACT is None: self.lrn_act = HF.tanh
			if OUT_ACT is None: self.out_act = HF.tanh
			if self.LOC_LRN_RULE == 'hpcat_ada': self.VAR_ADAPTIVE = True
		if self.LOC_LRN_RULE == 'hwta':
			if LRN_SIM is None:
				self.lrn_sim = HF.raised_cos_sim2d
				self.LRN_SIM_P = self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_P, 2.) # NB: In hwta the default lrn_sim is squared raised cosine
			if LRN_ACT is None: self.lrn_act = HF.identity
			if OUT_SIM is None: self.out_sim = HF.vector_proj2d
			if OUT_ACT is None: self.out_act = F.relu
			self.GATING = H.HebbianConv2d.GATE_BASE
			self.RECONSTR = H.HebbianConv2d.REC_QNT_SGN
			self.RED = H.HebbianConv2d.RED_W_AVG
		if self.LOC_LRN_RULE in ['ica', 'hica', 'ica_nrm', 'hica_nrm']:
			if LRN_ACT is None: self.lrn_act = HF.tanh
			if OUT_ACT is None: self.out_act = HF.tanh
			self.ACT_COMPLEMENT_INIT = self.HEBB_PARAM_DICT.get(PP.KEY_ACT_COMPLEMENT_INIT, None)
			self.ACT_COMPLEMENT_RATIO = self.HEBB_PARAM_DICT.get(PP.KEY_ACT_COMPLEMENT_RATIO, 0.)
			self.ACT_COMPLEMENT_ADAPT = self.HEBB_PARAM_DICT.get(PP.KEY_ACT_COMPLEMENT_ADAPT, None)
			self.ACT_COMPLEMENT_GRP = self.HEBB_PARAM_DICT.get(PP.KEY_ACT_COMPLEMENT_GRP, False)
			self.UPD_RULE = H.HebbianConv2d.UPD_ICA
			if self.LOC_LRN_RULE == 'hica': self.UPD_RULE = H.HebbianConv2d.UPD_HICA
			if self.LOC_LRN_RULE == 'ica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_ICA_NRM
			if self.LOC_LRN_RULE == 'hica_nrm': self.UPD_RULE = H.HebbianConv2d.UPD_HICA_NRM
			if self.LOC_LRN_RULE in ['ica_nrm', 'hica_nrm']: self.VAR_ADAPTIVE = True
			self.GATING = H.HebbianConv2d.GATE_BASE
		if self.LRN_SIM_EXP is not None: self.lrn_sim = HF.get_exp_sim(HF.get_affine_sim(self.lrn_sim, p=self.LRN_SIM_EXP), HF.get_pow_nc(utils.retrieve(self.HEBB_PARAM_DICT.get(PP.KEY_LRN_SIM_NC, None)), self.LRN_SIM_EXP))
		self.lrn_sim = HF.get_affine_sim(self.lrn_sim, self.LRN_SIM_B, self.LRN_SIM_S, self.LRN_SIM_P)
		self.lrn_act = HF.get_affine_act(self.lrn_act, self.LRN_ACT_SCALE_IN, self.LRN_ACT_SCALE_OUT, self.LRN_ACT_OFFSET_IN, self.LRN_ACT_OFFSET_OUT, self.LRN_ACT_P)
		if self.OUT_SIM_EXP is not None: self.out_sim = HF.get_exp_sim(HF.get_affine_sim(self.out_sim, p=self.OUT_SIM_EXP), HF.get_pow_nc(utils.retrieve(self.HEBB_PARAM_DICT.get(PP.KEY_OUT_SIM_NC, None)), self.OUT_SIM_EXP))
		self.out_sim = HF.get_affine_sim(self.out_sim, self.OUT_SIM_B, self.OUT_SIM_S, self.OUT_SIM_P)
		self.out_act = HF.get_affine_act(self.out_act, self.OUT_ACT_SCALE_IN, self.OUT_ACT_SCALE_OUT, self.OUT_ACT_OFFSET_IN, self.OUT_ACT_OFFSET_OUT, self.OUT_ACT_P)
		self.ALPHA_L = self.HEBB_PARAM_DICT.get(P.KEY_ALPHA_L, 1.)
		self.ALPHA_G = self.HEBB_PARAM_DICT.get(P.KEY_ALPHA_G, 0.)
	
	def create_hebb_layer(self, final=False, in_channels=1, out_channels=1, kernel_size=1, teacher_distrib=None):
		l = None
		
		if final:
			l = H.HebbianConv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=1,
				lrn_sim=HF.get_affine_sim(HF.raised_cos_sim2d, p=2),
				lrn_act=HF.identity,
				lrn_cmp=True,
				lrn_t=True,
				out_sim=HF.vector_proj2d if self.ALPHA_G == 0. else HF.kernel_mult2d,
				out_act=HF.identity,
				competitive=H.Competitive(),
				teacher_distrib=teacher_distrib,
				gating=H.HebbianConv2d.GATE_BASE,
				upd_rule=H.HebbianConv2d.UPD_RECONSTR if self.ALPHA_G == 0. else None,
				reconstruction=H.HebbianConv2d.REC_QNT_SGN,
				reduction=H.HebbianConv2d.RED_W_AVG,
				alpha_l=self.ALPHA_L,
				alpha_g=0. if self.ALPHA_G == 0. else 1.,
			)
		else:
			l = H.HebbianConv2d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=kernel_size,
				lrn_sim=self.lrn_sim,
				lrn_act=self.lrn_act,
				lrn_cmp=True,
				lrn_t=True,
				out_sim=self.out_sim,
				out_act=self.out_act,
				competitive=H.Competitive(out_size=HF.get_factors(out_channels), competitive_act=self.competitive_act, k=self.K),
				act_complement_init=self.ACT_COMPLEMENT_INIT,
				act_complement_ratio=self.ACT_COMPLEMENT_RATIO,
				act_complement_adapt=self.ACT_COMPLEMENT_ADAPT,
				act_complement_grp=self.ACT_COMPLEMENT_GRP,
				var_adaptive=self.VAR_ADAPTIVE,
				teacher_distrib=teacher_distrib if self.DEEP_TEACHER_SIGNAL else None,
				gating=self.GATING,
				upd_rule=self.UPD_RULE,
				reconstruction=self.RECONSTR,
				reduction=self.RED,
				alpha_l=self.ALPHA_L,
				alpha_g=self.ALPHA_G,
			)
		
		return l
	
	