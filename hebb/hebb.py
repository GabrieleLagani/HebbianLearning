import torch
import torch.nn as nn

from .functional import *
import params as P


# TODO:
#   - Competitive nonlinearities normalized to have max=1, Hebbian/anti-Hebbian competitive nonlinearity
#   - Remove border artifacts due to possible padding and introduce spatial decorrelation
#   - Post nonlinear demixer ica, mix ica and pca, reconstruction with bias
#   - Clustering with gauss nonlinearity integrated with vector projection similarity using weight vector as mean
#       encoding and bias for variance, so that it is possible to find aligned clusters with dot product-based similarity.
#   - Generalized parameter class for zeta mode as well as merged updates.
#   - Generalized normalization based on layer norm in addition to batch norm, Modified batch norm layer which computes stats only
#       adaptively, and not batch wise, and in the backward pass computes the gradients as if the stats were computed
#       batch wise. Same also for other adaptive params. Possibility to normalize with average of variances rather than
#       each feature with its own variance. Possibility to normalize with heuristics based on weights.
#   - Temporal competition
#   - Supervised Hebbian
#   - Add deferred update flag and local optimization inside module
#   - Other hebbian rules


# A generalized normalization layer
class GenNorm(nn.Module):
	
	def __init__(self, n, beta=0.1, eps=1e-9, affine=True):
		super(GenNorm, self).__init__()
		
		self.bias = nn.Parameter(torch.zeros(n).float(), requires_grad=True)
		self.register_buffer('running_mean', torch.zeros(n).float())
		#self.running_mean = nn.Parameter(torch.zeros(n).float(), requires_grad=True)
		self.weight = nn.Parameter(torch.ones(n).float(), requires_grad=True)
		self.register_buffer('running_var', torch.ones(n).float())
		#self.running_var = nn.Parameter(torch.ones(n).float(), requires_grad=True)
		
		self.beta = beta
		self.eps = eps
		self.affine = affine
	
	def track(self, x):
		mean = x.mean(dim=(0, 2))
		var = x.var(dim=(0, 2))
		self.running_mean = self.running_mean + self.beta * (mean - self.running_mean)
		self.running_var = self.running_var + self.beta * (var - self.running_var)
	
	def normalize(self, x):
		res = (x - self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + self.eps)
		if self.affine: res = res * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
		return res
		
	def forward(self, x):
		orig_size = x.size()
		x = x.view(x.size(0), x.size(1), -1) # Batch dim, channel dim, input/window dim
		with torch.no_grad(): self.track(x)
		return self.normalize(x).view(orig_size)
	
	#def parameters(self, recurse: bool = ...):
	#	return [self.weight, self.bias]


# A layer of competitive activation
class Competitive(nn.Module):
	# Types of random abstention strategies
	HARD_RAND_ABST = 'hard_rand_abst'
	SOFT_RAND_ABST = 'soft_rand_abst'
	
	# Types of LFB kernels
	LFB_GAUSS = 'lfb_gauss'
	LFB_DoG = 'lfb_DoG'
	LFB_EXP = 'lfb_exp'
	LFB_DoE = 'lfb_DoE'
	
	# Types of adaptation mechanisms for the k parameter
	ADA_K_MODE_STD = 'ada_k_mode_std'
	ADA_K_MODE_LOG = 'ada_k_mode_log'
	ADA_K_MODE_SHIFT = 'ada_k_mode_shift'
	
	
	def __init__(self,
	             out_size=None,
	             competitive_act=None,
	             k=1,
	             lrn_k=False,
	             ada_k=None,
	             random_abstention=None,
	             y_gating=False,
	             lfb_y_gating=False,
	             lfb_value=None,
	             lfb_sigma=None,
	             lfb_tau=1000,
	             beta=.1):
		# Default gives always all 1s in output, i.e. all winners (trivial competition). Competitive(lfb_y_gating=True) gives identity mapping.
		super(Competitive, self).__init__()
		# Enable/disable features such as random abstention, competitive learning, lateral feedback
		self.competitive_act = competitive_act
		self.competitive = self.competitive_act is not None
		self.beta = beta
		self.k = k if not lrn_k else nn.Parameter(torch.tensor(float(k)), requires_grad=True)
		if ada_k not in [None, self.ADA_K_MODE_STD, self.ADA_K_MODE_LOG, self.ADA_K_MODE_SHIFT]:
			raise ValueError("Invalid value for argument ada_k: " + str(ada_k))
		self.ada_k = ada_k
		self.trk = nn.BatchNorm2d(1, momentum=self.beta, affine=False)
		if random_abstention not in [None, self.SOFT_RAND_ABST, self.HARD_RAND_ABST]:
			raise ValueError("Invalid value for argument random_abstention: " + str(random_abstention))
		self.random_abstention = random_abstention
		self.random_abstention_on = self.competitive and self.random_abstention is not None
		self.y_gating = y_gating
		self.lfb_y_gating = lfb_y_gating
		self.lfb_value = lfb_value
		self.lfb_on = self.lfb_value is not None and self.lfb_value != 0
		
		# Initialize output size, which is necessary only when random abstention or lfb is enabled
		self.out_size = None
		self.out_channels = None
		if self.random_abstention_on or self.lfb_on:
			if out_size is None:
				raise ValueError("Invalid value for argument out_size: " + str(out_size) + " when random abstention or lfb is provided")
			if hasattr(out_size, '__len__') and len(out_size) > 3:
				raise ValueError("Too many dimensions for argument out_size: " + str(out_size) + " (up to 3 allowed)")
			out_size_list = [out_size] if not hasattr(out_size, '__len__') else out_size
			self.out_size = torch.tensor(out_size_list)
			self.out_channels = self.out_size.prod().item()
		
		# Set parameters related to the lateral feedback feature
		if self.lfb_on:
			# Prepare the variables to generate the kernel that will be used to apply lateral feedback
			map_radius = (self.out_size - 1) // 2
			lfb_sigma = map_radius.max().item() if lfb_sigma is None else lfb_sigma
			x = torch.abs(torch.arange(0, self.out_size[0].item()) - map_radius[0])
			for i in range(1, self.out_size.size(0)):
				x_new = torch.abs(torch.arange(0, self.out_size[i].item()) - map_radius[i])
				for j in range(i): x_new = x_new.unsqueeze(j)
				x = torch.max(x.unsqueeze(-1), x_new)  # max gives L_infinity distance, sum would give L_1 distance, root_p(sum x^p) for L_p
			# Store the kernel that will be used to apply lateral feedback in a registered buffer
			if lfb_value == self.LFB_EXP or lfb_value == self.LFB_DoE:
				self.register_buffer('lfb_kernel', torch.exp(-x.float() / lfb_sigma))
			if lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG:
				self.register_buffer('lfb_kernel', torch.exp(-x.pow(2).float() / (2 * (lfb_sigma ** 2))))
			else: # lfb_value is a number
				if type(lfb_value) is not int and type(lfb_value) is not float:
					raise ValueError("Invalid value for argument lfb_value: " + str(lfb_value))
				self.register_buffer('lfb_kernel', (x == 0).float())
				x[x == 0] = lfb_value
			# Padding that will pad the inputs before applying the lfb kernel
			pad_pre = map_radius.unsqueeze(1)
			pad_post = (self.out_size - 1 - map_radius).unsqueeze(1)
			self.pad = list(torch.cat((pad_pre, pad_post), dim=1).flip(0).view(-1))
			# LFB kernel shrinking parameter
			self.gamma = torch.exp(torch.log(torch.tensor(lfb_sigma).float()) / lfb_tau).item() if lfb_tau is not None else None
			if (lfb_value == self.LFB_GAUSS or lfb_value == self.LFB_DoG) and self.gamma is not None: self.gamma = self.gamma ** 2
		else: self.register_buffer('lfb_kernel', None)
		
		# Init variables for statistics collection
		if self.random_abstention_on:
			self.register_buffer('victories_count', torch.zeros(self.out_channels).float())
		else: self.register_buffer('victories_count', None)
	
	def get_k(self, scores, lrn=False):
		k = self.k
		if self.training and lrn: # Track stats
			if self.ada_k == self.ADA_K_MODE_LOG: _ = self.trk(torch.log(scores).view(-1, 1, 1, 1))
			else:
				if self.ada_k is not None: #NB: this condition will be removed when we implement more efficient tracking (currently it is slow)
					_ = self.trk(scores.view(-1, 1, 1, 1))
		# Compute adaptive k
		if self.ada_k in [self.ADA_K_MODE_STD, self.ADA_K_MODE_LOG]:
			k = k * self.trk.running_var**0.5
		if self.ada_k in [self.ADA_K_MODE_SHIFT]:
			k = -self.trk.running_mean + k * self.trk.running_var**0.5
		return k
	
	def forward(self, y, t=None, lrn=False):
		# Random abstention
		scores = y
		if self.random_abstention_on:
			abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
			if self.random_abstention == self.SOFT_RAND_ABST: scores = y * abst_prob.unsqueeze(0)
			if self.random_abstention == self.HARD_RAND_ABST: scores = y * (torch.rand_like(abst_prob) >= abst_prob).float().unsqueeze(0)
		
		# Competition. The returned winner_mask is a bitmap telling where a neuron won and where one lost.
		if self.competitive:
			winner_mask = self.competitive_act(scores, self.get_k(scores, lrn=lrn), t)
			if lrn and self.random_abstention_on and self.training:  # Update statistics if using random abstension
				winner_mask_sum = winner_mask.sum(0)  # Number of inputs over which a neuron won
				self.victories_count += winner_mask_sum
				self.victories_count -= self.victories_count.min().item()
		else: winner_mask = torch.ones_like(y)
		
		# Apply winner_mask gating by the output if necessary
		if self.y_gating: winner_mask = winner_mask * y
		
		# Lateral feedback
		if self.lfb_on:
			lfb_kernel = self.lfb_kernel
			if self.lfb_value == self.LFB_DoG or self.lfb_value == self.LFB_DoE: lfb_kernel = 2 * lfb_kernel - lfb_kernel.pow(0.5)  # Difference of Gaussians/Exponentials (mexican hat shaped function)
			lfb_in = F.pad(winner_mask.view(-1, *self.out_size), self.pad)
			if self.out_size.size(0) == 1: lfb_out = torch.conv1d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			elif self.out_size.size(0) == 2: lfb_out = torch.conv2d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			else: lfb_out = torch.conv3d(lfb_in.unsqueeze(1), lfb_kernel.unsqueeze(0).unsqueeze(1))
			lfb_out = lfb_out.clamp(-1, 1).view_as(y)
		else: lfb_out = winner_mask
		
		# Apply lfb gating by output if necessary
		if self.lfb_y_gating: lfb_out = lfb_out * y
		
		# LFB kernel shrinking schedule
		if lrn and self.lfb_on and self.gamma is not None and self.training: self.lfb_kernel = self.lfb_kernel.pow(self.gamma)
		
		return lfb_out


# This module represents a layer of convolutional neurons that are trained with Hebbian algorithms
class HebbianConv2d(nn.Module):
	# s = sim(w, x), y = act(s) z = competition(y) -- e.g.: s = w^T x, y = f(s), z = soft-WTA(y)
	
	# Type of gating term
	GATE_BASE = 'gate_base'  # g = z
	GATE_HEBB = 'gate_hebb'  # g = z * y
	GATE_DIFF = 'gate_diff'  # g = z - y
	GATE_SMAX = 'gate_smx'  # g = z - softmax(y)
	
	# Type of recombination terms
	RECOMB_BASE = 'recomb_base'  # r = 1
	RECOMB_Y = 'recomb_y'  # r = y
	RECOMB_Z = 'recomb_z'  # r = z
	
	# Type of reconstruction scheme
	REC_QNT = 'rec_qnt'  # reconstr = w
	REC_QNT_SGN = 'rec_qnt_sgn'  # reconstr = sign(g) * w
	REC_LIN_CMB = 'rec_lin_cmb'  # reconstr = sum_i r_i w_i
	
	# Type of update step
	UPD_RECONSTR = 'upd_reconstr' # delta_w = alpha * g * (x - reconstr)
	UPD_ICA = 'upd_ica' # delta w_i = g_i (w_i - z_i rs^T W)
	UPD_HICA = 'upd_hica' # delta w_i = g_i (w_i - z_i sum_(k=1..i) rs_k w_k)
	UPD_ICA_NRM = 'upd_ica_nrm' # delta w_i = = g_i (w_i w_i^T - I) z_i rs^T W
	UPD_HICA_NRM = 'upd_hica_nrm' # delta w_i = g_i (w_i w_i^T - I) z_i sum_(k=1..i) rs_k w_k
	
	# ICA rule
	# Delta W = (I - f(s) s^T) W
	# Delta W_i = sum_k (I - f(s) s^T)_ik W_k = sum_k (delta_ik - f(s_i) s_k) W_k
	# = delta_ii W_i - f(s_i) sum_k s_k W_k = W_i - f(s_i) sum_k s_k W_k  = W_i - f(s_i) s^T W
	# ICA with normalization: estimate sigma dinamically
	# f(s) = (log(p(s/sigma)))' = (1/sigma) * ( p'(s/sigma)/p(s/sigma) ) = (1/sigma) * phi(s/sigma)
	# where phi(s) = p'(s)/p(s)
	# rewrite: Delta W = (I - f(s) s^T) W = (I - (1/sigma^2) * sigma^2 f(s) s^T) W  -
	# [Note that Delta W ~ (sigma^2 - g(s) s^T) W -- where g(s) = sigma^s f(s)]
	# Delta W = (I - f(s) s^T) W
	# Delta W_i = sum_k (I - f(s) s^T)_ik W_k = sum_k (delta_ik - f(s_i) s_k) W_k
	# = W_i - f(s_i) sum_k s_k W_k
	# Normalization
	# w_new = (w + eta Delta w) / |w + eta Delta w|
	# |w + eta Delta w|^2 = w^2 (1 + 2 eta w^T Delta w) + o(eta^2)
	# (|w + eta Delta w|^2)^-1/2 = |w| (1 - eta w^T Delta w) + o(eta^2)
	# w_new = (w + eta Delta w)(w^-2 (1 - eta w^T Delta w) + o(eta^2))
	# = (w + eta (Delta w - (w^t Delta w) w) w^-2 + o(eta^2)
	# Assuming w normalized --> w^-2 = 1
	# w_new = w + eta (Delta w - (w^T Delta w) w) + o(eta^2)
	# Delta w corrected = Delta w - (w^T Delta w) w --> x y - y^2 w for oja
	# In case of adaptive variance ICA
	# Delta w_i corrected = w_i - f(s_i) sum_k s_k w_k - w_i(w_i^2 - f(s_i) sum_k s_k (w_k w_i^T))      ** NB: w_i is row vector
	# = w_i - w_i -f(s_i) sum_k s_k (w_k - w_i (w_k w_i^T)) = -f(s_i) sum_(k!=i) s_k (w_k - w_i (w_k w_i^T))
	# = -f(s_i) sum_k s_k (w_k - w_k w_i^T w_i) = -f(s_i) sum_k s_k w_k (I - w_i^T w_i) = -f(s_i) s^T W (I - w_i w_i^T)
	# = f(s_i) s^T W w_i^T w_i  - f(s_i) s^T W = f(s_i) s^T W (w_i^T w_i - I)
	
	# Type of update reduction scheme
	RED_AVG = 'red_avg'  # average
	RED_W_AVG = 'red_w_avg'  # weighted average
	
	# Types of bias update scheme
	BIAS_MODE_BASE = 'bias_mode_base'
	BIAS_MODE_TARG = 'bias_mode_targ'
	BIAS_MODE_STD = 'bias_mode_std'
	BIAS_MODE_PERC = 'bias_mode_perc'
	BIAS_MODE_EXP = 'bias_mode_exp'
	BIAS_MODE_VALUE = 'bias_mode_value'
	
	# Activation complement modes - activation complement transforms nonlinearities for some neuron into x - the nonlinearity.
	# This can be used during ica training to have different nonlinearities for super-gaussian and sub-gaussian variables.
	ACT_COMPLEMENT_INIT_RAND = 'act_complement_init_rand'
	ACT_COMPLEMENT_INIT_SPLT = 'act_complement_init_splt'
	ACT_COMPLEMENT_INIT_ALT  = 'act_complement_init_alt'
	
	# Types of activation inversion adaptive update scheme
	ACT_COMPLEMENT_ADAPT_KRT = 'act_complement_adapt_krt'
	ACT_COMPLEMENT_ADAPT_STB = 'act_complement_adapt_stb'
	
	# Modes for affine parameters
	ZETA_MODE_CONST = 'zeta_mode_const'
	ZETA_MODE_PARAM = 'zeta_mode_param'
	ZETA_MODE_VEC = 'zeta_mode_vec'
	ZETA_MODE_MAT = 'zeta_mode_mat'
	
	# Modes for teacher signal distribution
	TEACHER_DISTRIB_SOFT_GROUPS = 'teacher_distrib_soft_groups'
	
	def __init__(self,
	             in_channels,
	             out_channels,
	             kernel_size,
	             weight_init=None,
	             weight_init_nrm=False,
	             weight_zeta_mode=None,
	             weight_zeta=0.,
	             lrn_sim=kernel_mult2d,
	             lrn_act=identity,
	             lrn_cmp=False,
	             lrn_t=False,
	             lrn_bias=False,
	             out_sim=kernel_mult2d,
	             out_act=identity,
	             out_cmp=False,
	             out_t=False,
	             out_bias=False,
	             competitive=None,
	             act_complement_init=None,
	             act_complement_ratio=0,
	             act_complement_adapt=None,
	             act_complement_grp=False,
	             act_complement_affine=False,
	             teacher_distrib=None,
	             gating=GATE_HEBB,
	             upd_rule=UPD_RECONSTR,
	             adaptive_step=True,
	             y_prime_gating=False,
	             z_prime_gating=False,
	             z_gating=True,
	             recombination=RECOMB_Y,
	             reconstruction=REC_LIN_CMB,
	             reduction=RED_AVG,
	             bias_init=None,
	             bias_mode=None,
	             bias_agg=False,
	             bias_target=0,
	             bias_gating=None,
	             bias_var_gating=False,
	             bias_zeta_mode=None,
	             bias_zeta=0.,
	             var_adaptive=False,
	             var_affine=False,
	             conserve_var=True,
	             alpha_l=1,
	             alpha_g=0,
	             alpha_bias_l=1,
	             alpha_bias_g=0,
	             beta=.1,):
		super(HebbianConv2d, self).__init__()
		# Init weights
		if hasattr(kernel_size, '__len__') and len(kernel_size) == 1: kernel_size = kernel_size[0]
		if not hasattr(kernel_size, '__len__'): kernel_size = [kernel_size, kernel_size]
		self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
		if not callable(weight_init) and weight_init is not None:
			raise ValueError("Argument weight_init must be callable or None")
		if weight_init is None: weight_init = weight_init_std # Default initialization
		self.weight = weight_init(self.weight)
		if weight_zeta_mode not in [None, self.ZETA_MODE_CONST, self.ZETA_MODE_PARAM, self.ZETA_MODE_VEC, self.ZETA_MODE_MAT]:
			raise ValueError("Invalid value for argument weight_zeta_mode: " + str(weight_zeta_mode))
		self.weight_zeta_mode = weight_zeta_mode
		if self.weight_zeta_mode is not None:
			self.weight_ada = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
			self.weight_ada = weight_init(self.weight_ada)
			self.weight_zeta = weight_zeta
			if self.weight_zeta_mode == self.ZETA_MODE_PARAM: self.weight_zeta = nn.Parameter(torch.tensor(weight_zeta).float(), requires_grad=True)
			if self.weight_zeta_mode == self.ZETA_MODE_VEC: self.weight_zeta = nn.Parameter(weight_zeta * torch.ones(out_channels).float().view(-1, 1, 1, 1), requires_grad=True)
			if self.weight_zeta_mode == self.ZETA_MODE_MAT: self.weight_zeta = nn.Parameter(weight_zeta * torch.ones_like(self.weight_ada).float(), requires_grad=True)
			self.weight = self.weight / (1 + (1 + self.weight_zeta))
			self.weight_ada = self.weight_ada / (1 + (1 + self.weight_zeta))
		else:
			self.register_parameter('weight_ada', None)
			self.register_parameter('weight_zeta', None)
		if weight_init_nrm: # Normalize weights
			w_norm = self.get_weight().view(self.weight.size(0), -1).norm(dim=1, p=2).view(-1, 1, 1, 1)
			self.weight = self.weight / w_norm
			if self.weight_ada is not None: self.weight_ada = self.weight_ada / w_norm
		self.adaptive_step = adaptive_step
		
		# Alpha is the constant which determines the trade off between global and local updates
		self.alpha_l = alpha_l
		self.alpha_g = alpha_g
		self.alpha_bias_l = alpha_bias_l
		self.alpha_bias_g = alpha_bias_g
		
		# Set similarity and activation functions
		self.lrn_sim = lrn_sim
		self.lrn_act = lrn_act
		self.lrn_cmp = lrn_cmp
		self.lrn_t = lrn_t
		self.lrn_bias = lrn_bias
		self.out_sim = out_sim
		self.out_act = out_act
		self.out_cmp = out_cmp
		self.out_t = out_t
		self.out_bias = out_bias
		self.competitive = competitive # None gives identity mapping
		if self.competitive.out_channels is not None and self.competitive.out_channels != out_channels:
			raise ValueError("Argument out_channels: " + str(out_channels) + " and competitive.out_channels: " + str(self.competitive.out_channels) + " must match")
		if act_complement_init not in [None, self.ACT_COMPLEMENT_INIT_RAND, self.ACT_COMPLEMENT_INIT_SPLT, self.ACT_COMPLEMENT_INIT_ALT]:
			raise ValueError("Invalid value for argument act_complement_init: " + str(act_complement_init))
		if act_complement_adapt not in [None, self.ACT_COMPLEMENT_ADAPT_STB, self.ACT_COMPLEMENT_ADAPT_KRT]:
			raise ValueError("Invalid value for argument act_complement_adapt: " + str(act_complement_adapt))
		if act_complement_ratio > 1.0:
			raise ValueError("Invalid value for argument act_complement_ratio: " + str(act_complement_ratio) + " (required float <= 1.0)")
		kappa = None
		self.act_complement_from_idx = out_channels
		if act_complement_init == self.ACT_COMPLEMENT_INIT_RAND:
			kappa = (torch.rand(out_channels) < act_complement_ratio).float()
		if act_complement_init == self.ACT_COMPLEMENT_INIT_SPLT:
			kappa = torch.zeros(out_channels).float()
			self.act_complement_from_idx = out_channels - int(round(act_complement_ratio * out_channels))
			if self.act_complement_from_idx < out_channels: kappa[self.act_complement_from_idx:] = 1.
		if act_complement_init == self.ACT_COMPLEMENT_INIT_ALT:
			if act_complement_ratio <= 0.5:
				kappa = torch.zeros(out_channels).float()
				if act_complement_ratio > 0:
					n = int(round(1/act_complement_ratio))
					idx = [n * (i + 1) - 1 for i in range(out_channels//n)]
					kappa[idx] = 1.
			else:
				kappa = torch.ones(out_channels).float()
				if act_complement_ratio < 1:
					n = int(round(1/(1 - act_complement_ratio)))
					idx = [n * i for i in range(out_channels//n)]
					kappa[idx] = 0.
		self.register_buffer('kappa', kappa)
		self.act_complement_adapt = act_complement_adapt
		self.register_buffer('m2', torch.ones(out_channels).float() if self.act_complement_adapt is not None else None)
		self.register_buffer('m4', (3 * torch.ones(out_channels).float()) if self.act_complement_adapt is not None else None)
		self.register_buffer('rho', (3 * torch.ones(out_channels).float()) if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_STB else None)
		self.act_complement_grp = act_complement_grp and (self.act_complement_from_idx < out_channels) and (self.act_complement_adapt is None)
		self.kappa_affine = act_complement_affine
		if self.kappa_affine: self.kappa_trainable = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
		else: self.register_parameter('kappa_trainable', None)
		
		# Init bias
		if bias_mode not in [None, self.BIAS_MODE_BASE, self.BIAS_MODE_TARG, self.BIAS_MODE_STD, self.BIAS_MODE_PERC, self.BIAS_MODE_EXP, self.BIAS_MODE_VALUE]:
			raise ValueError("Invalid value for argument bias_mode: " + str(bias_mode))
		if not (bias_init is None or isinstance(bias_init, int) or isinstance(bias_init, float) or (isinstance(bias_init, str) and bias_mode == self.BIAS_MODE_VALUE) or callable(bias_init)):
			raise ValueError("Invalid value for argument bias_init: " + str(bias_init) + " when argument bias_mode is: " + str(bias_mode))
		if bias_gating not in [None, self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
			raise ValueError("Invalid value for argument bias_gating: " + str(bias_gating))
		bias = None
		if isinstance(bias_init, int) or isinstance(bias_init, float) or isinstance(bias_init, str): bias = bias_init
		if callable(bias_init): bias = bias_init(self.weight)
		self.bias_mode = bias_mode
		if self.bias_mode == self.BIAS_MODE_VALUE: self.bias = bias
		else: self.bias = nn.Parameter(bias * torch.ones(out_channels).float(), requires_grad=True) if bias is not None else self.register_parameter('bias', None)
		self.bias_agg = bias_agg
		self.bias_target = bias_target
		self.bias_gating = bias_gating
		self.bias_var_gating = bias_var_gating
		self.using_updatable_bias = self.bias is not None and self.bias_mode != self.BIAS_MODE_VALUE
		self.using_adaptive_bias = self.alpha_bias_l != 0 and self.using_updatable_bias and self.bias_mode is not None and self.lrn_bias
		if bias_zeta_mode not in [None, self.ZETA_MODE_CONST, self.ZETA_MODE_PARAM, self.ZETA_MODE_VEC, self.ZETA_MODE_MAT]:
			raise ValueError("Invalid value for argument bias_zeta_mode: " + str(bias_zeta_mode))
		self.bias_zeta_mode = bias_zeta_mode
		if self.bias_zeta_mode is not None:
			if not self.using_updatable_bias:
				raise ValueError("Invalid argument bias_zeta_mode when bias in non-numeric")
			self.bias_ada = nn.Parameter(bias * torch.ones(out_channels).float(), requires_grad=True)
			self.bias_zeta = bias_zeta
			if self.bias_zeta_mode == self.ZETA_MODE_PARAM: self.bias_zeta = nn.Parameter(torch.tensor(bias_zeta).float(), requires_grad=True)
			if self.bias_zeta_mode in [self.ZETA_MODE_VEC, self.ZETA_MODE_MAT]: self.bias_zeta = nn.Parameter(bias_zeta * torch.ones(out_channels).float(), requires_grad=True)
			self.bias = self.bias / (1 + (1 + self.bias_zeta))
			self.bias_ada = self.bias_ada / (1 + (1 + self.bias_zeta))
		else:
			self.register_parameter('bias_ada', None)
			self.register_parameter('bias_zeta', None)
		
		# Learning rule
		self.teacher_signal = None  # Teacher signal for supervised training
		if not (teacher_distrib in [None, self.TEACHER_DISTRIB_SOFT_GROUPS] or (isinstance(teacher_distrib, int) and teacher_distrib >= 0)):
			raise ValueError("Invalid value for argument teacher_distrib: " + str(teacher_distrib))
		self.teacher_distrib = teacher_distrib
		if gating not in [self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
			raise ValueError("Invalid value for argument gating: " + str(gating))
		self.gating = gating
		if recombination not in [None, self.RECOMB_BASE, self.RECOMB_Y, self.RECOMB_Z]:
			raise ValueError("Invalid value for argument recombination: " + str(recombination))
		self.recombination = recombination
		if reconstruction not in [None, self.REC_QNT, self.REC_QNT_SGN, self.REC_LIN_CMB]:
			raise ValueError("Invalid value for argument reconstruction: " + str(reconstruction))
		self.reconstruction = reconstruction
		if upd_rule not in [None, self.UPD_RECONSTR, self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
			raise ValueError("Invalid value for argument upd_rule: " + str(upd_rule))
		self.upd_rule = upd_rule
		self.using_adaptive_weight = self.alpha_l != 0 and self.upd_rule is not None
		self.y_prime_gating = y_prime_gating
		self.z_prime_gating = z_prime_gating
		self.z_gating = z_gating
		if reduction not in [self.RED_AVG, self.RED_W_AVG]:
			raise ValueError("Invalid value for argument reductioin: " + str(reduction))
		self.reduction = reduction
		
		# Adaptive variance normalization
		self.beta = beta # Beta is the time constant for running stats tracking
		self.trk = nn.BatchNorm2d(out_channels, momentum=self.beta, affine=var_affine)
		self.var_adaptive = var_adaptive
		self.conserve_var = conserve_var
		
		# Variables where the weight updates are stored
		self.delta_w = None
		self.delta_bias = None
		
	def forward(self, x):
		if self.training: self.compute_update(x)
		out = self.out_sim(x, self.get_weight(), self.get_bias() if self.out_bias else None)
		out_shape = out.size()
		out = self.apply_act(out)
		t = self.teacher_signal if self.out_t else None
		if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(out)
		out = out.permute(0, 2, 3, 1).contiguous().view(-1, out.size(1))
		if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
		out = self.competitive(out, t) if self.competitive is not None and self.out_cmp else out
		out = out * t if t is not None else out
		return out.view(out_shape[0], out_shape[2], out_shape[3], out_shape[1]).permute(0, 3, 1, 2).contiguous()
	
	def get_weight(self):
		weight = self.weight
		if self.weight_ada is not None: weight = weight + (1 + self.weight_zeta) * self.weight_ada
		return weight
	
	def get_bias(self):
		bias = self.bias
		if self.bias_ada is not None: bias = bias + (1 + self.bias_zeta) * self.bias_ada
		return bias.mean() if self.bias_agg else self.bias
	
	def apply_act(self, s, lrn=False, cpl=True):
		s_bar = s
		
		# Normalize before activation function, if necessary
		if lrn: _ = self.trk(s) # Track stats
		if self.var_adaptive:
			s_bar = (s - self.trk.running_mean.view(1, -1, 1, 1)) / ((self.trk.running_var.view(1, -1, 1, 1) + self.trk.eps) ** 0.5)
			if self.trk.affine: s_bar = s_bar * self.trk.weight.view(1, -1, 1, 1) + self.trk.bias.view(1, -1, 1, 1)
			s_bar = s_bar + self.trk.running_mean.view(1, -1, 1, 1) # Restore original mean and normalize variance only
		
		# Apply activation function
		y = self.lrn_act(s_bar) if lrn else self.out_act(s_bar)
		# Apply activation complement, if necessary
		if cpl:
			kappa = self.kappa.view(1, -1, 1, 1) if self.kappa is not None else 0.
			if self.kappa_affine: kappa = kappa + self.kappa_trainable.view(1, -1, 1, 1)
			y = kappa * s_bar - (2 * kappa - 1) * y
		
		# Restore original variance information, if necessary
		if self.conserve_var and self.var_adaptive: y = y * (self.trk.running_var.view(1, -1, 1, 1) + self.trk.eps) ** 0.5
		
		return y
	
	def set_teacher_signal(self, t):
		if t is not None:
			if self.teacher_distrib is None or self.teacher_distrib == 0:
				self.teacher_signal = None
			
			elif self.teacher_distrib == self.TEACHER_DISTRIB_SOFT_GROUPS:
				raise NotImplemented
			
			else: # self.teacher_distrib is a positive integer
				if self.teacher_distrib * t.size(1) <= self.weight.size(0):
					self.teacher_signal = torch.cat((
						torch.ones(t.size(0), self.weight.size(0) - self.teacher_distrib * t.size(1), device=t.device),
						t.view(t.size(0), t.size(1), 1).repeat(1, 1, self.teacher_distrib).view(t.size(0), -1)
					), dim=1)
				else: self.teacher_signal = None
		
		else: self.teacher_signal = t
	
	def compute_update(self, x):
		# Store previous gradient computation flag and disable gradient computation before computing update
		prev_grad_enabled = torch.is_grad_enabled()
		torch.set_grad_enabled(False)
		
		if self.using_adaptive_weight or self.using_adaptive_bias or self.var_adaptive or self.act_complement_adapt is not None:
			# Compute activation states for the layer: s, y, y'
			s = self.lrn_sim(x, self.get_weight(), self.get_bias() if self.lrn_bias else None) # Compute similarity metric between inputs and weights
			# Compute y and also y' if y' gating is required
			if self.y_prime_gating:
				torch.set_grad_enabled(True) # Gradient enabling to compute derivative y'
				s.requires_grad = True
			y = self.apply_act(s, lrn=True)
			y_prime = torch.ones_like(s)
			if self.y_prime_gating:
				y.backward(torch.ones_like(y), retain_graph=prev_grad_enabled, create_graph=prev_grad_enabled)
				y_prime = s.grad
				s.grad = None
				torch.set_grad_enabled(False)
			
			# Track higher order moments
			if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_KRT:
				# Update statistics and determine kappa
				self.m2 = (1 - self.beta) * self.m2 + self.beta * s.pow(2).mean(dim=(0, 2, 3))
				self.m4 = (1 - self.beta) * self.m4 + self.beta * s.pow(4).mean(dim=(0, 2, 3))
				self.kappa = ((self.m4 - 3 * self.m2 ** 2) < 0).float()
			if self.act_complement_adapt == self.ACT_COMPLEMENT_ADAPT_STB:
				# compute y' uncomplemented, which is needed for this type of adaptation
				torch.set_grad_enabled(True) # Gradient enabling to compute derivative y' uncomplemented
				s.requires_grad = True
				y_uncpl = self.apply_act(s, cpl=False)
				y_uncpl.backward(torch.ones_like(y_uncpl), retain_graph=prev_grad_enabled, create_graph=prev_grad_enabled)
				y_uncpl_prime = s.grad
				s.grad = None
				torch.set_grad_enabled(False)
				# Update statistics and determine kappa
				self.m2 = (1 - self.beta) * self.m2 + self.beta * s.pow(2).mean(dim=(0, 2, 3))
				self.m4 = (1 - self.beta) * self.m4 + self.beta * (s * y_uncpl).mean(dim=(0, 2, 3))
				self.rho = (1 - self.beta) * self.rho + self.beta * y_uncpl_prime.mean(dim=(0, 2, 3))
				self.kappa = ((self.m4 - self.m2 * self.rho) < 0).float()
			
			if self.using_adaptive_weight or self.using_adaptive_bias:
				# Prepare the necessary tensors and set them in the correct shape
				t = self.teacher_signal if self.lrn_t else None
				if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(s)
				s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				y_prime = y_prime.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
				x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)
				
				# Run competition, and also derivative w.r.t. competitive nonlinearity, if necessary.
				if self.z_prime_gating:
					torch.set_grad_enabled(True) # Gradient enabling to compute derivative z'
					s.requires_grad = True
				z = self.competitive(y, t, lrn=True) if self.competitive is not None and self.lrn_cmp else y
				z_prime = torch.ones_like(y)
				if self.z_prime_gating:
					z.backward(torch.ones_like(z), retain_graph=prev_grad_enabled, create_graph=prev_grad_enabled)
					z_prime = y.grad
					y.grad = None
					torch.set_grad_enabled(False)
				
				if self.using_adaptive_weight:
					# Compute step modulation coefficient
					g = (z if self.z_gating else torch.ones_like(z)) * (t if t is not None else 1.) # GATE_BASE
					if self.gating == self.GATE_HEBB: g = g * y
					if self.gating == self.GATE_DIFF: g = g - y
					if self.gating == self.GATE_SMAX: g = g - torch.softmax(y, dim=1)
					if self.y_prime_gating: g = y_prime * g
					if self.z_prime_gating: g = z_prime * g
					
					# Compute recombination coefficients
					r = torch.ones_like(z) * (t if t is not None else 1.) # RECOMB_BASE
					if self.recombination == self.RECOMB_Y: r = r * y
					if self.recombination == self.RECOMB_Z: r = r * z
					
					# Compute the coefficients for update reduction/aggregation over the batch.
					# Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
					# update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
					# determine the length of each step (RED_W_AVG), or the unweighted average (RED_AVG).
					c = 1/g.size(0)
					if self.reduction == self.RED_W_AVG:
						g_sum = g.abs().sum(dim=0, keepdim=True)
						g_sum = g_sum + (g_sum == 0).float()  # Prevent divisions by zero
						c = g.abs()/g_sum
					
					# Compute delta_w
					delta_w_agg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
					for grp in range(2): # repeat the computation for the two neuron groups using complementary nonlinearities
						if grp == 1 and not self.act_complement_grp: break
						grp_slice = slice(self.weight.size(0))
						if self.act_complement_grp: grp_slice = slice(self.act_complement_from_idx) if grp == 0 else slice(self.act_complement_from_idx, self.weight.size(0))
						w = (self.weight_ada if self.weight_ada is not None else self.weight).view(1, self.weight.size(0), -1)[:, grp_slice, :]
						x_bar = None
						grlw = None
						sw = None
						
						if P.HEBB_FASTHEBB:
							if not P.HEBB_REORDMULT:
								if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_ICA_NRM: # Computation of s^T * W for ICA
									for i in range((w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
										start = i * P.HEBB_UPD_GRP
										end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
										w_i = w[:, start:end, :]
										s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
										r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
										sw = (r_i * s_i).view(r_i.size(0), -1).matmul(w_i.view(w_i.size(1), -1)).unsqueeze(1) + (sw if sw is not None else 0.)
								
							for i in range((w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
								start = i * P.HEBB_UPD_GRP
								end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
								w_i = w[:, start:end, :]
								s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
								y_i = y[:, grp_slice].unsqueeze(2)[:, start:end, :]
								z_i = z[:, grp_slice].unsqueeze(2)[:, start:end, :]
								g_i = g[:, grp_slice].unsqueeze(2)[:, start:end, :]
								r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
								c_i = c[:, grp_slice].unsqueeze(2)[:, start:end, :] if isinstance(c, torch.Tensor) else c
								gc = g_i * c_i
								
								# Compute update step
								delta_w_i = torch.zeros_like(w_i)
								if self.upd_rule == self.UPD_RECONSTR:
									delta_w_i = gc.view(gc.size(0), -1).t().matmul(x_unf.view(x_unf.size(0), -1))
									# Compute reconstr based on the type of reconstruction
									if self.reconstruction == self.REC_QNT: delta_w_i = delta_w_i - gc.sum(0) * w_i.view(w_i.size(1), -1)
									elif self.reconstruction == self.REC_QNT_SGN: delta_w_i = delta_w_i - (gc * g_i.sign()).sum(0) * w_i.view(w_i.size(1), -1)
									elif self.reconstruction == self.REC_LIN_CMB:
										if P.HEBB_REORDMULT:
											l_i = (torch.arange(w.size(1), device=w.device).unsqueeze(0).repeat(w_i.size(1), 1) <= torch.arange(start, end, device=w.device).unsqueeze(1)).float()
											grlw = (gc.view(gc.size(0), -1).t().matmul(r[:, grp_slice]) * l_i).matmul(w.view(w.size(1), -1))
											delta_w_i = delta_w_i - grlw
										else:
											x_bar = torch.cumsum(r_i * w_i, dim=1) + (x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
											delta_w_i = delta_w_i - gc.permute(1, 2, 0).matmul(x_bar.permute(1, 0, 2)).view(w_i.size(1), -1)
								if self.upd_rule in [self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
									if P.HEBB_REORDMULT:
										if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
											l_i = (torch.arange(w.size(1), device=w.device).unsqueeze(0).repeat(w_i.size(1), 1) <= torch.arange(start, end, device=w.device).unsqueeze(1)).float()
											gzsw = ((gc * z_i).view(gc.size(0), -1).t().matmul(r[:, grp_slice] * s[:, grp_slice]) * l_i).matmul(w.view(w.size(1), -1))
										else:
											gzsw = (gc * z_i).view(gc.size(0), -1).t().matmul(r[:, grp_slice] * s[:, grp_slice]).matmul(w.view(w.size(1), -1))
									else:
										if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
											sw = torch.cumsum((r_i * s_i) * w_i, dim=1) + (sw[:, -1, :].unsqueeze(1) if sw is not None else 0.)
										gzsw = (gc * z_i * sw).sum(0)
									if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_HICA:
										delta_w_i = gc.sum(0) * w_i.view(w_i.size(1), -1) - gzsw
									if self.upd_rule == self.UPD_ICA_NRM or self.upd_rule == self.UPD_HICA_NRM:
										delta_w_i = (gzsw * w_i.view(w_i.size(1), -1)).sum(dim=1, keepdim=True) * w_i.view(w_i.size(1), -1) - gzsw
										
								# Store aggregated update in buffer
								delta_w_agg[grp_slice, :][start:end, :] = delta_w_i
								
						else:
							if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_ICA_NRM: # Computation of s^T * W for ICA
								for i in range((w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
									start = i * P.HEBB_UPD_GRP
									end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
									w_i = w[:, start:end, :]
									s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
									r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
									sw = (r_i * s_i * w_i).sum(dim=1, keepdim=True) + (sw if sw is not None else 0.)
							
							for i in range((w.size(1) // P.HEBB_UPD_GRP) + (1 if w.size(1) % P.HEBB_UPD_GRP != 0 else 0)):
								start = i * P.HEBB_UPD_GRP
								end = min((i + 1) * P.HEBB_UPD_GRP, w.size(1))
								w_i = w[:, start:end, :]
								s_i = s[:, grp_slice].unsqueeze(2)[:, start:end, :]
								y_i = y[:, grp_slice].unsqueeze(2)[:, start:end, :]
								z_i = z[:, grp_slice].unsqueeze(2)[:, start:end, :]
								g_i = g[:, grp_slice].unsqueeze(2)[:, start:end, :]
								c_i = c[:, grp_slice].unsqueeze(2)[:, start:end, :] if isinstance(c, torch.Tensor) else c
								
								# Compute update step
								delta_w_i = torch.zeros_like(w_i)
								if self.upd_rule == self.UPD_RECONSTR:
									# Compute reconstr based on the type of reconstruction
									if self.reconstruction == self.REC_QNT: x_bar = w_i
									elif self.reconstruction == self.REC_QNT_SGN: x_bar = g_i.sign() * w_i
									elif self.reconstruction == self.REC_LIN_CMB: x_bar = torch.cumsum(r_i * w_i, dim=1) + (x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
									else: x_bar = 0.
									delta_w_i = g_i * (x_unf - x_bar)
								if self.upd_rule in [self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
									if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
										sw = torch.cumsum((r_i * s_i) * w_i, dim=1) + (sw[:, -1, :].unsqueeze(1) if sw is not None else 0.)
									zsw = (z_i * sw)
									if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_HICA:
										delta_w_i = g_i * (w_i - zsw)
									if self.upd_rule == self.UPD_ICA_NRM or self.upd_rule == self.UPD_HICA_NRM:
										delta_w_i = g_i * ((zsw * w_i).sum(dim=2, keepdim=True) * w_i - zsw)
								
								# Aggregate updates over batch
								delta_w_agg[grp_slice, :][start:end, :] = (delta_w_i * c_i).sum(0)
					
					# Store delta
					self.delta_w = delta_w_agg.view_as(self.weight)
					# Layer size-based adaptive step (not necessary if we are using WTA-type learning)
					if self.adaptive_step and not (self.upd_rule == self.UPD_RECONSTR and self.reconstruction in [self.REC_QNT, self.REC_QNT_SGN]):
						self.delta_w = self.delta_w / ((self.weight.size(1) * self.weight.size(2) * self.weight.size(3))**0.5)
				
				if self.using_adaptive_bias:
					# Compute step modulation coefficient
					g = (z if self.z_gating else torch.ones_like(z)) * (t if t is not None else 1.) # GATE_BASE
					b = (self.bias_ada if self.bias_ada is not None else self.bias).view(1, -1)
					if self.bias_gating == self.GATE_HEBB: g = g * y
					if self.bias_gating == self.GATE_DIFF: g = g - y
					if self.bias_gating == self.GATE_SMAX: g = g - torch.softmax(y, dim=1)
					if self.bias_gating is None: g = 1.
					if self.y_prime_gating: g = y_prime * g
					if self.z_prime_gating: g = z_prime * g
					if self.bias_var_gating: g = g * s * (torch.log(s) / b.view(1, -1))
					
					# Compute the coefficients for update reduction/aggregation over the batch.
					c = 1/g.size(0)
					if self.reduction == self.RED_W_AVG:
						g_sum = g.abs().sum(dim=0, keepdim=True)
						g_sum = g_sum + (g_sum == 0).float()  # Prevent divisions by zero
						c = g.abs()/g_sum
					
					# Compute Delta bias
					delta_bias = torch.zeros_like(self.bias).unsqueeze(0)
					if self.bias_mode == self.BIAS_MODE_BASE:
						delta_bias = g
					if self.bias_mode == self.BIAS_MODE_TARG:
						# NB: self.bias_target is the target mean that we want the bias to achieve
						delta_bias = g * (self.bias_target - s)
					if self.bias_mode == self.BIAS_MODE_STD:
						# NB: self.bias_target is the number of std devs away from the mean that we want the bias to achieve
						delta_bias = g * (self.bias_target * (self.trk.running_var ** 0.5).view(1, -1) - s)
					if self.bias_mode == self.BIAS_MODE_PERC:
						# NB: self.bias target is the target percentile of samples above zero that we want the similarity function to achieve
						delta_bias = g * (self.bias_target - (s < 0).float())
					if self.bias_mode == self.BIAS_MODE_EXP:
						# NB: self.bias_target is the number of std devs that we want the scale of the similarity function to achieve
						delta_bias = g * b * (1 - (-torch.log(s)) * self.bias_target)
					
					# Aggregate updates over the batch and store delta
					self.delta_bias = c.t().unsqueeze(1).matmul(delta_bias.t().unsqueeze(2)).view_as(self.bias)
					# Layer size-based adaptive step (not necessary if we are using WTA-type learning)
					if self.adaptive_step and not (self.upd_rule == self.UPD_RECONSTR and self.reconstruction in [self.REC_QNT, self.REC_QNT_SGN]):
						self.delta_bias = self.delta_bias / ((self.weight.size(1) * self.weight.size(2) * self.weight.size(3))**0.5)
		
		# Restore gradient computation
		torch.set_grad_enabled(prev_grad_enabled)
		
	# Takes local update from self.delta_w and self.delta_bias, global update from self.weight.grad and self.bias.grad,
	# and combines them using the parameter alpha.
	def local_update(self):
		if self.delta_w is not None or self.weight.grad is not None:
			if self.weight_ada is None:
				# NB: self.delta_w has a minus sign in front because the optimizer will take update steps in the opposite direction.
				self.weight.grad = self.alpha_l * (-self.delta_w if self.delta_w is not None else torch.zeros_like(self.weight).float()) + self.alpha_g * (self.weight.grad if self.weight.grad is not None else torch.zeros_like(self.weight).float())
			else:
				self.weight_ada.grad = self.alpha_l * (-self.delta_w if self.delta_w is not None else torch.zeros_like(self.weight).float())
				self.weight.grad = self.alpha_g * (self.weight.grad if self.weight.grad is not None else torch.zeros_like(self.weight).float())
			self.delta_w = None
		if self.using_updatable_bias and (self.delta_bias is not None or self.bias.grad is not None):
			if self.bias_ada is None:
				# NB: self.delta_bias has a minus sign in front because the optimizer will take update steps in the opposite direction.
				self.bias.grad = self.alpha_bias_l * (-self.delta_bias if self.delta_bias is not None else torch.zeros_like(self.bias).float()) + self.alpha_bias_g * (self.bias.grad if self.bias.grad is not None else torch.zeros_like(self.bias).float())
			else:
				self.bias_ada.grad = self.alpha_bias_l * (-self.delta_bias if self.delta_bias is not None else torch.zeros_like(self.bias).float())
				self.bias.grad = self.alpha_bias_g * (self.bias.grad if self.bias.grad is not None else torch.zeros_like(self.bias).float())
			self.delta_bias = None

