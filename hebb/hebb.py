import torch.nn as nn

from .functional import *
import params as P


# TODO: Implement Hebbian update directly on gpu.
# TODO: Add other functionalities to Hebbian module
#   - Recurrent computation and lca
#   - Lateral decorrelation
#   - Trisigma wta



class Competitive(nn.Module):
	# Types of random abstention strategies
	HARD_RAND_ABST = 'hard_rand_abst'
	SOFT_RAND_ABST = 'soft_rand_abst'
	
	# Types of LFB kernels
	LFB_GAUSS = 'lfb_gauss'
	LFB_DoG = 'lfb_DoG'
	LFB_EXP = 'lfb_exp'
	LFB_DoE = 'lfb_DoE'
	
	
	def __init__(self,
	             out_size=None,
	             competitive_act=None,
	             k=1,
	             lrn_k=False,
	             random_abstention=None,
	             y_gating=False,
	             lfb_y_gating=False,
	             lfb_value=None,
	             lfb_sigma=None,
	             lfb_tau=1000):
		super(Competitive, self).__init__()
		# Enable/disable features such as random abstention, competitive learning, lateral feedback
		self.competitive_act = competitive_act
		self.competitive = self.competitive_act is not None
		self.k = k if not lrn_k else nn.Parameter(torch.tensor(float(k)), requires_grad=True)
		if self.competitive and random_abstention not in [None, self.SOFT_RAND_ABST, self.HARD_RAND_ABST]:
			raise ValueError("Invalid value for argument random_abstention: " + str(random_abstention))
		self.random_abstention = random_abstention
		self.random_abstention_on = self.competitive and self.random_abstention is not None
		self.y_gating = y_gating
		self.lfb_y_gating = lfb_y_gating
		self.lfb_on = lfb_value is not None and lfb_value != 0
		self.lfb_value = lfb_value
		
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
	
	def forward(self, y, t=None):
		# Random abstention
		scores = y
		if self.random_abstention_on:
			abst_prob = self.victories_count / (self.victories_count.max() + y.size(0) / y.size(1)).clamp(1)
			if self.random_abstention == self.SOFT_RAND_ABST: scores = y * abst_prob.unsqueeze(0)
			if self.random_abstention == self.HARD_RAND_ABST: scores = y * (torch.rand_like(abst_prob, device=y.device) >= abst_prob).float().unsqueeze(0)
		
		# Competition. The returned winner_mask is a bitmap telling where a neuron won and where one lost.
		if self.competitive:
			if t is not None: scores = scores * t # When competition is on, teacher signal is used to drive competition, if provided
			winner_mask = self.competitive_act(scores, self.k)
			if self.random_abstention_on and self.training:  # Update statistics if using random abstension
				winner_mask_sum = winner_mask.sum(0)  # Number of inputs over which a neuron won
				self.victories_count += winner_mask_sum
				self.victories_count -= self.victories_count.min().item()
		else: winner_mask = torch.ones_like(y, device=y.device)
		
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
		
		# When competition is off, the teacher signal is used to gate the lfb output, if provided
		if not self.competitive and t is not None: lfb_out = lfb_out * t
		
		# LFB kernel shrinking schedule
		if self.lfb_on and self.gamma is not None and self.training: self.lfb_kernel = self.lfb_kernel.pow(self.gamma)
		
		return lfb_out

# This module represents a layer of convolutional neurons that are trained with Hebbian algorithms
class HebbianConv2d(nn.Module):
	# s = sim(w, x), y = act(s) -- e.g.: s = w^T x, y = f(s)
	
	# Types of weight initialization schemes
	INIT_BASE = 'init_base'
	INIT_NORM = 'init_norm'
	
	# Type of gating term
	GATE_BASE = 'gate_base'  # r = lfb
	GATE_HEBB = 'gate_hebb'  # r = lfb * y
	GATE_DIFF = 'gate_diff'  # r = lfb - y
	GATE_SMAX = 'gate_smx'  # r = lfb - softmax(y)
	
	# Type of reconstruction scheme
	REC_QNT = 'rec_qnt'  # reconstr = w
	REC_QNT_SGN = 'rec_qnt_sgn'  # reconstr = sign(lfb) * w
	REC_LIN_CMB = 'rec_lin_cmb'  # reconstr = sum_i r_i w_i
	
	# Type of update step
	UPD_RECONSTR = 'upd_reconstr' # delta_w = alpha * r * (x - reconstr)
	UPD_ICA = 'upd_ica' # delta w_i = r_i (w_i - y_i rs^T W) # NB: the term r acts as gating
	UPD_HICA = 'upd_hica' # delta w_i = r_i (w_i - y_i sum_(k=1..i) rs_k w_k)
	UPD_ICA_NRM = 'upd_ica_nrm' # delta w_i = = r_i (w_i w_i^T - I) y_i rs^T W
	UPD_HICA_NRM = 'upd_hica_nrm' # delta w_i = r_i (w_i w_i^T - I) y_i sum_(k=1..i) rs_k w_k
	
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
	# w_new = (w + eta Delta w) / |w + eta Delta w|^2
	# |w + eta Delta w|^2 = w^2 (1 + eta w^T Delta w w^-2) + o(eta^2)
	# |w + eta Delta w|^-2 = w^-2 (1 - eta w^T Delta w) + o(eta^2)
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
	
	# Types of bias initialization schemes
	BIAS_INIT_ZEROS = 'bias_init_zeros'
	BIAS_INIT_VAR_ONES = 'bias_init_var_ones'
	BIAS_INIT_VAR_DIMS = 'bias_init_var_dims'
	
	# Types of bias update scheme
	BIAS_MODE_BASE = 'bias_mode_base'
	BIAS_MODE_HEBB = 'bias_mode_hebb'
	BIAS_MODE_VALUE = 'bias_mode_value'
	
	# Activation inversion modes - activation inversion transforms nonlinearities for some neuron into x - the nonlinearity.
	# This can be used during ica training to have different nonlinearities for super-gaussian and sub-gaussian variables.
	ACT_COMPLEMENT_INIT_RAND = 'act_complement_init_rand'
	ACT_COMPLEMENT_INIT_SPLT = 'act_complement_init_splt'
	ACT_COMPLEMENT_INIT_ALT  = 'act_complement_init_alt'
	
	# Types of activation inversion adaptive update scheme
	ACT_COMPLEMENT_ADAPT_KRT = 'act_complement_adapt_krt'
	ACT_COMPLEMENT_ADAPT_STB = 'act_complement_adapt_stb'
	
	
	def __init__(self,
	             in_channels,
	             out_channels,
	             kernel_size,
	             weight_init=INIT_BASE,
	             lrn_sim=kernel_mult2d,
	             lrn_act=identity,
	             lrn_cmp=None,
	             out_sim=kernel_mult2d,
	             out_act=identity,
	             out_cmp=None,
	             act_complement_init=None,
	             act_complement_ratio=0,
	             act_complement_adapt=None,
	             act_complement_grp=False,
	             act_complement_affine=False,
	             gating=GATE_HEBB,
	             upd_rule=UPD_RECONSTR,
	             y_prime_gating=False,
	             reconstruction=REC_LIN_CMB,
	             reduction=RED_AVG,
	             bias_init=None,
	             bias_mode=None,
	             bias_target=0,
	             bias_gating=None,
	             var_adaptive=False,
	             var_affine=False,
	             lamb=1.,
	             conserve_var=True,
	             alpha=1.,
	             alpha_bias=1.,
	             beta=.1):
		super(HebbianConv2d, self).__init__()
		# Init weights
		if weight_init not in [self.INIT_BASE, self.INIT_NORM]:
			raise ValueError("Invalid value for argument weight_init: " + str(weight_init))
		if hasattr(kernel_size, '__len__') and len(kernel_size) == 1: kernel_size = kernel_size[0]
		if not hasattr(kernel_size, '__len__'): kernel_size = [kernel_size, kernel_size]
		stdv = 1 / (in_channels * kernel_size[0] * kernel_size[1]) ** 0.5
		self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1]), requires_grad=True)
		nn.init.uniform_(self.weight, -stdv, stdv)  # Same initialization used by default pytorch conv modules (the one from the paper "Efficient Backprop, LeCun")
		if weight_init == self.INIT_NORM: self.weight = self.weight / self.weight.view(self.weight.size(0), -1).norm(dim=1, p=2).view(-1, 1, 1, 1)  # normalize weights
		
		# Init bias
		if bias_mode not in [None, self.BIAS_MODE_BASE, self.BIAS_MODE_HEBB, self.BIAS_MODE_VALUE]:
			raise ValueError("Invalid value for argument bias_mode: " + str(bias_mode))
		if bias_init not in [None, self.BIAS_INIT_ZEROS, self.BIAS_INIT_VAR_ONES, self.BIAS_INIT_VAR_DIMS] and bias_mode != self.BIAS_MODE_VALUE:
			raise ValueError("Invalid value for argument bias_init: " + str(bias_init) + " when argument bias_mode is: " + str(bias_mode))
		if bias_gating not in [None, self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
			raise ValueError("Invalid value for argument bias_gating: " + str(bias_gating))
		bias = None
		if bias_init == self.BIAS_INIT_ZEROS: bias = torch.zeros(out_channels).float()
		if bias_init == self.BIAS_INIT_VAR_ONES: bias = torch.ones(out_channels).float()
		if bias_init == self.BIAS_INIT_VAR_DIMS: bias = utils.shape2size(tuple(self.weight[0].size())) * torch.ones(out_channels).float()
		self.bias_mode = bias_mode
		if self.bias_mode == self.BIAS_MODE_VALUE: self.bias = bias_init
		else: self.bias = nn.Parameter(bias, requires_grad=True) if bias is not None else self.register_parameter('bias', None)
		self.bias_target = bias_target
		self.bias_gating = bias_gating
		self.using_adaptive_bias = self.bias is not None and self.bias_mode is not None and self.bias_mode != self.BIAS_MODE_VALUE
		
		# Set similarity and activation functions
		self.lrn_sim = lrn_sim
		self.lrn_act = lrn_act
		self.lrn_cmp = lrn_cmp if lrn_cmp is not None else Competitive(lfb_y_gating=True) # None gives identity mapping
		if self.lrn_cmp.out_channels is not None and self.lrn_cmp.out_channels != out_channels:
			raise ValueError("Argument out_channels: " + str(out_channels) + " and lrn_cmp.out_channels: " + str(self.lrn_cmp.out_channels) + " must match")
		self.out_sim = out_sim
		self.out_act = out_act
		self.out_cmp = out_cmp if out_cmp is not None else Competitive(lfb_y_gating=True) # None gives identity mapping
		if self.out_cmp.out_channels is not None and self.out_cmp.out_channels != out_channels:
			raise ValueError("Argument out_channels: " + str(out_channels) + " and out_cmp.out_channels: " + str(self.out_cmp.out_channels) + " must match")
		if act_complement_init not in [None, self.ACT_COMPLEMENT_INIT_RAND, self.ACT_COMPLEMENT_INIT_SPLT, self.ACT_COMPLEMENT_INIT_ALT]:
			raise ValueError("Invalid value for argument act_complement_init: " + str(act_complement_init))
		if act_complement_adapt not in [None, self.ACT_COMPLEMENT_ADAPT_STB, self.ACT_COMPLEMENT_ADAPT_KRT]:
			raise ValueError("Invalid value for argument act_complement_adapt: " + str(act_complement_adapt))
		if act_complement_ratio > 1.0:
			raise ValueError("Invalid value for argument act_complement_ration: " + str(act_complement_ratio) + " (required float < 1.0)")
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
		self.kappa_trainable = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True) if self.kappa_affine else self.register_parameter('kappa_trainable', None)
		
		# Learning rule
		self.teacher_signal = None  # Teacher signal for supervised training
		if gating not in [self.GATE_BASE, self.GATE_HEBB, self.GATE_DIFF, self.GATE_SMAX]:
			raise ValueError("Invalid value for argument gating: " + str(gating))
		self.gating = gating
		if reconstruction not in [None, self.REC_QNT, self.REC_QNT_SGN, self.REC_LIN_CMB]:
			raise ValueError("Invalid value for argument reconstruction: " + str(reconstruction))
		self.reconstruction = reconstruction
		if upd_rule not in [self.UPD_RECONSTR, self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
			raise ValueError("Invalid value for argument upd_rule: " + str(upd_rule))
		self.upd_rule = upd_rule
		self.y_prime_gating = y_prime_gating
		if reduction not in [self.RED_AVG, self.RED_W_AVG]:
			raise ValueError("Invalid value for argument reductioin: " + str(reduction))
		self.reduction = reduction
		
		# Alpha is the constant which determines the trade off between global and local updates
		self.alpha = alpha
		self.alpha_bias = alpha_bias
		
		# Adaptive variance normalization
		self.beta = beta # Beta is the time constant for running stats tracking
		self.var_adaptive = var_adaptive
		self.var_nrm = nn.BatchNorm2d(out_channels, momentum=self.beta, affine=var_affine) if self.var_adaptive else None
		self.lamb = lamb # lambda is an additional spread parameter for nonlinearities. It determines, for instance, the sparsity measure when a shrinking/clamping nonlinearity is used
		self.conserve_var = conserve_var
		
		# Variables where the weight updates are stored
		self.delta_w = None
		self.delta_bias = None
	
	def set_teacher_signal(self, t):
		self.teacher_signal = t
		
	def forward(self, x):
		if self.training: self.compute_update(x)
		return self.out_cmp(self.apply_act(self.out_sim(x, self.weight, self.bias)))
	
	def apply_act(self, s, lrn=False, cpl=True):
		s_bar = s
		# Normalize before activation function, if necessary
		if self.var_adaptive:
			if lrn: _ = self.var_nrm(s) # Track stats
			s_bar = (s - self.var_nrm.running_mean.view(1, -1, 1, 1)) / ((self.var_nrm.running_var.view(1, -1, 1, 1) + self.var_nrm.eps) ** 0.5)
			if self.var_nrm.affine: s_bar = s_bar * self.var_nrm.weight.view(1, -1, 1, 1) + self.var_nrm.bias.view(1, -1, 1, 1)
			s_bar = s_bar + self.var_nrm.running_mean.view(1, -1, 1, 1) # Restore original mean and normalize variance only
		# Scale width before activation function
		s_bar = s_bar / self.lamb
		
		# Apply activation function
		y = self.lrn_act(s_bar) if lrn else self.out_act(s_bar)
		# Apply activation complement, if necessary
		if cpl:
			kappa = self.kappa.view(1, -1, 1, 1) if self.kappa is not None else 0.
			if self.kappa_affine: kappa = kappa + self.kappa_trainable.view(1, -1, 1, 1)
			y = kappa * s_bar - (2 * kappa - 1) * y
		
		# Restore original variance information, if necessary
		if self.conserve_var and self.var_adaptive: y = y * (self.var_nrm.running_var.view(1, -1, 1, 1) + self.var_nrm.eps)**0.5
		
		return y
	
	def compute_update(self, x):
		# Store previous gradient computation flag and disable gradient computation before computing update
		prev_grad_enabled = torch.is_grad_enabled()
		torch.set_grad_enabled(False)
		
		if self.alpha != 0 or (self.alpha_bias != 0 and self.using_adaptive_bias) or self.var_adaptive or self.act_complement_adapt is not None:
			# Compute activation states for the layer: s, y, y'
			s = self.lrn_sim(x, self.weight, self.bias) # Compute similarity metric between inputs and weights
			# Compute y and also y' if y' gating is required
			if self.y_prime_gating:
				torch.set_grad_enabled(True) # Gradient enabling to compute derivative y'
				s.requires_grad = True
			y = self.apply_act(s, lrn=True)
			y_prime = torch.ones_like(s)
			if self.y_prime_gating:
				y.backward(torch.ones_like(y), retain_graph=prev_grad_enabled)
				y_prime = s.grad.clone().detach()
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
				y_uncpl.backward(torch.ones_like(y_uncpl), retain_graph=prev_grad_enabled)
				y_uncpl_prime = s.grad.clone().detach()
				s.grad = None
				torch.set_grad_enabled(False)
				# Update statistics and determine kappa
				self.m2 = (1 - self.beta) * self.m2 + self.beta * s.pow(2).mean(dim=(0, 2, 3))
				self.m4 = (1 - self.beta) * self.m4 + self.beta * (s * y_uncpl).mean(dim=(0, 2, 3))
				self.rho = (1 - self.beta) * self.rho + self.beta * y_uncpl_prime.mean(dim=(0, 2, 3))
				self.kappa = ((self.m4 - self.m2 * self.rho) < 0).float()
			
			if self.alpha != 0 or (self.alpha_bias != 0 and self.using_adaptive_bias):
				# Prepare the necessary tensors and set them in the correct shape
				t = self.teacher_signal
				if t is not None: t = t.unsqueeze(2).unsqueeze(3) * torch.ones_like(s, device=s.device)
				s = s.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				y = y.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				y_prime = y_prime.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				if t is not None: t = t.permute(0, 2, 3, 1).contiguous().view(-1, self.weight.size(0))
				x_unf = unfold_map2d(x, self.weight.size(2), self.weight.size(3))
				x_unf = x_unf.permute(0, 2, 3, 1, 4).contiguous().view(s.size(0), 1, -1)
				
				# Run competition
				lfb_out = self.lrn_cmp(y, t)
				
				if self.alpha != 0:
					# Compute step modulation coefficient
					r = lfb_out  # GATE_BASE
					if self.gating == self.GATE_HEBB: r = r * y
					if self.gating == self.GATE_DIFF: r = r - y
					if self.gating == self.GATE_SMAX: r = r - torch.softmax(y, dim=1)
					if self.y_prime_gating: r = y_prime * r
					r_abs = r.abs()
					r_sign = r.sign()
				
					# Compute delta_w (serialized version for computation of delta_w using less memory)
					delta_w_avg = torch.zeros_like(self.weight.view(self.weight.size(0), -1))
					for grp in range(2): # repeat the computation for the two neuron groups using complementary nonlinearities
						if grp == 1 and not self.act_complement_grp: break
						grp_slice = slice(self.weight.size(0))
						if self.act_complement_grp: grp_slice = slice(self.act_complement_from_idx) if grp == 0 else slice(self.act_complement_from_idx, self.weight.size(0))
						w = self.weight.view(1, self.weight.size(0), -1)[:, grp_slice, :]
						x_bar = None
						sw = None
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
							r_i = r[:, grp_slice].unsqueeze(2)[:, start:end, :]
							r_abs_i = r_abs[:, grp_slice].unsqueeze(2)[:, start:end, :]
							r_sign_i = r_sign[:, grp_slice].unsqueeze(2)[:, start:end, :]
							# Compute update step
							delta_w_i = torch.zeros_like(w_i)
							if self.upd_rule == self.UPD_RECONSTR:
								# Compute reconstr based on the type of reconstruction
								if self.reconstruction == self.REC_QNT: x_bar = w_i
								if self.reconstruction == self.REC_QNT_SGN: x_bar = r_sign_i * w_i
								if self.reconstruction == self.REC_LIN_CMB: x_bar = torch.cumsum(r_i * w_i, dim=1) + (x_bar[:, -1, :].unsqueeze(1) if x_bar is not None else 0.)
								x_bar = x_bar if x_bar is not None else 0.
								delta_w_i = r_i * (x_unf - x_bar)
							if self.upd_rule in [self.UPD_ICA, self.UPD_HICA, self.UPD_ICA_NRM, self.UPD_HICA_NRM]:
								if self.upd_rule == self.UPD_HICA or self.upd_rule == self.UPD_HICA_NRM:
									sw = torch.cumsum((r_i * s_i) * w_i, dim=1) + (sw[:, -1, :].unsqueeze(1) if sw is not None else 0.)
								ysw = (y_i * sw)
								if self.upd_rule == self.UPD_ICA or self.upd_rule == self.UPD_HICA:
									delta_w_i = r_i * (w_i - ysw)
								if self.upd_rule == self.UPD_ICA_NRM or self.upd_rule == self.UPD_HICA_NRM:
									delta_w_i = r_i * ((ysw * w_i).sum(dim=2, keepdim=True) * w_i - ysw)
							# Since we use batches of inputs, we need to aggregate the different update steps of each kernel in a unique
							# update. We do this by taking the weighted average of the steps, the weights being the r coefficients that
							# determine the length of each step (RED_W_AVG), or the unweighted average (RED_AVG).
							if self.reduction == self.RED_W_AVG:
								r_sum = r_abs_i.sum(0)
								r_sum = r_sum + (r_sum == 0).float()  # Prevent divisions by zero
								delta_w_avg[grp_slice, :][start:end, :] = (delta_w_i * r_abs_i).sum(0) / r_sum
							else: # RED_AVG
								delta_w_avg[grp_slice, :][start:end, :] = delta_w_i.mean(dim=0)
					
					# Apply delta
					self.delta_w = delta_w_avg.view_as(self.weight)
				
				if self.alpha_bias != 0 and self.using_adaptive_bias:
					# Compute step modulation coefficient
					r = lfb_out  # GATE_BASE
					if self.bias_gating == self.GATE_HEBB: r = r * y
					if self.bias_gating == self.GATE_DIFF: r = r - y
					if self.bias_gating == self.GATE_SMAX: r = r - torch.softmax(y, dim=1)
					if self.bias_gating is None: r = 1.
					if self.y_prime_gating: r = y_prime * r
					r_abs = r.abs()
					r_sign = r.sign()
					
					# Compute Delta bias
					delta_bias = torch.zeros_like(self.bias).unsqueeze(0)
					delta_bias_avg = torch.zeros_like(self.bias)
					if self.bias_mode == self.BIAS_MODE_BASE:
						delta_bias = r
					if self.bias_mode == self.BIAS_MODE_HEBB:
						delta_bias = r * -(s - self.bias_target)
					# Aggregate bias updates by averaging
					if self.reduction == self.RED_W_AVG:
						r_sum = r_abs.sum(0)
						r_sum = r_sum + (r_sum == 0).float()  # Prevent divisions by zero
						delta_bias_avg = (delta_bias * r_abs).sum(0) / r_sum
					else: # RED_AVG
						delta_bias_avg = delta_bias.mean(dim=0)
					
					# Apply delta
					self.delta_bias = delta_bias_avg.view_as(self.bias)
		
		# Restore gradient computation
		torch.set_grad_enabled(prev_grad_enabled)
		
	# Takes local update from self.delta_w and self.delta_bias, global update from self.weight.grad and self.bias.grad,
	# and combines them using the parameter alpha.
	def local_update(self):
		if self.alpha != 0:
			# NB: self.delta_w has a minus sign in front because the optimizer will take update steps in the opposite direction.
			self.weight.grad = self.alpha * (-self.delta_w) + (1 - self.alpha) * (self.weight.grad if self.weight.grad is not None else 0.)
		if self.alpha_bias != 0 and self.using_adaptive_bias:
			# NB: self.delta_bias has a minus sign in front because the optimizer will take update steps in the opposite direction.
			self.bias.grad = self.alpha_bias * (-self.delta_bias) + (1 - self.alpha_bias) * (self.bias.grad if self.bias.grad is not None else 0.)

