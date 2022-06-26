import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from neurolab import utils


# Euclidean distance between vectors
def dist(x, y):
	return (x.norm(p=2, dim=1).pow(2).view(-1, 1) + y.norm(p=2, dim=1).pow(2).view(1, -1) - 2 * x.matmul(y.t())) ** 0.5


# Standard weight initialization method, default used in pytorch conv modules (the one from the paper "Efficient Backprop, LeCun")
def weight_init_std(w):
	stdv = 1 / (utils.shape2size(w[0].size())) ** 0.5
	torch.nn.init.uniform_(w, -stdv, stdv)
	return w


# Apply unfold operation to input in order to prepare it to be processed against a sliding kernel whose shape
# is passed as argument.
def unfold_map2d(input, kernel_height, kernel_width):
	# Before performing an operation between an x and a sliding kernel we need to unfold the x, i.e. the
	# windows on which the kernel is going to be applied are extracted and set apart. For this purpose, the kernel
	# shape is passed as argument to the operation. The single extracted windows are reshaped by the unfold operation
	# to rank 1 vectors. The output of F.unfold(x, (kernel_height, kernel_width)).transpose(1, 2) is a
	# tensor structured as follows: the first dimension is the batch dimension; the second dimension is the slide
	# dimension, i.e. each element is a window extracted at a different offset (and reshaped to a rank 1 vector);
	# the third dimension is a scalar within said vector.
	inp_unf = F.unfold(input, (kernel_height, kernel_width)).transpose(1, 2)
	# Now we need to reshape our tensors to the actual shape that we want in output, which is the following: the
	# first dimension is the batch dimension, the second dimension is the output channels dimension, the third and
	# fourth are height and width dimensions (obtained by splitting the former third dimension, the slide dimension,
	# representing a linear offset within the x map, into two new dimensions representing height and width), the
	# fifth is the window components dimension, corresponding to the elements of a window extracted from the x with
	# the unfold operation (reshaped to rank 1 vectors). The resulting tensor is then returned.
	inp_unf = inp_unf.view(
		input.size(0),  # Batch dimension
		1,  # Output channels dimension
		input.size(2) - kernel_height + 1,  # Height dimension
		input.size(3) - kernel_width + 1,  # Width dimension
		-1  # Filter/window dimension
	)
	return inp_unf

# Custom vectorial function representing sum of an input with a sliding kernel, just like convolution is multiplication
# by a sliding kernel (as an analogy think convolution as a kernel_mult2d)
def kernel_sum2d(input, kernel):
	# In order to perform the sum with the sliding kernel we first need to unfold the x. The resulting tensor will
	# have the following structure: the first dimension is the batch dimension, the second dimension is the output
	# channels dimension, the third and fourth are height and width dimensions, the fifth is the filter/window
	# components dimension, corresponding to the elements of a window extracted from the x with the unfold
	# operation and equivalently to the elements of a filter (reshaped to rank 1 vectors)
	inp_unf = unfold_map2d(input, kernel.size(2), kernel.size(3))
	# At this point the two tensors can be summed. The kernel is reshaped by unsqueezing singleton dimensions along
	# the batch dimension and the height and width dimensions. By exploiting broadcasting, it happens that the inp_unf
	# tensor is broadcast over the output channels dimension (since its shape along this dimension is 1) and therefore
	# it is automatically processed against the different filters of the kernel. In the same way, the kernel is
	# broadcast along the first dimension (and thus automatically processed against the different inputs along
	# the batch dimension) and along the third and fourth dimensions (and thus automatically processed against
	# different windows extracted from the image at different height and width offsets).
	out = inp_unf + kernel.view(1, kernel.size(0), 1, 1, -1)
	return out

# Compute product between input and sliding kernel
def kernel_mult2d(x, w, bias=None):
	if bias is None: bias = 0.
	if isinstance(bias, torch.Tensor): bias = bias.view(1, -1, 1, 1)
	return F.conv2d(x, w) + bias

# Projection of input on weight vectors
def vector_proj2d(x, w, bias=None):
	# Compute scalar product with sliding kernel
	prod = kernel_mult2d(x, w)
	# Divide by the norm of the weight vector to obtain the projection
	norm_w = torch.norm(w.view(w.size(0), -1), p=2, dim=1).view(1, -1, 1, 1)
	norm_w = norm_w + (norm_w == 0).float()  # Prevent divisions by zero
	if bias is None: bias = 0.
	if isinstance(bias, torch.Tensor): bias = bias.view(1, -1, 1, 1)
	return prod / norm_w + bias

# Cosine similarity between input and a sliding kernel
def cos_sim2d(x, w, bias=None):
	proj = vector_proj2d(x, w)
	# Divide by the norm of the x to obtain the cosine similarity
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	norm_x = torch.norm(x_unf, p=2, dim=4)
	norm_x += (norm_x == 0).float()  # Prevent divisions by zero
	if bias is None: bias = 0.
	if isinstance(bias, torch.Tensor): bias = bias.view(1, -1, 1, 1)
	return proj / norm_x + bias

# Angular similarity between input and a sliding kernel
def ang_sim2d(x, w, bias=None):
	s = 1 - 2 * torch.acos(cos_sim2d(x, w))/math.pi
	if bias is None: bias = 0.
	if isinstance(bias, torch.Tensor): bias = bias.view(1, -1, 1, 1)
	return s + bias

# Cosine similarity remapped to 0, 1
def raised_cos_sim2d(x, w, bias=None):
	nc = nc_raised_cos_sim
	s = (cos_sim2d(x, w) + 1) / 2
	b = None
	if bias is None: b = 1.
	if bias == 'nc': b = nc(w)
	if isinstance(bias, torch.Tensor): b = bias.view(1, -1, 1, 1)
	if b is None: raise ValueError("Invalid value for parameter bias: " + str(bias))
	return s.pow(b)
# Corresponding normalization condition functions for heuristic variance determination
def nc_raised_cos_sim(w):
	return 1/(-math.log( (1 + torch.cos( 1 / (w.size(0) ** (1 / (utils.shape2size(tuple(w[0].size()))))) )) / 2 ))

# Angular similarity remapped to 0, 1
def raised_ang_sim2d(x, w, bias=None):
	nc = nc_raised_ang_sim
	s = (ang_sim2d(x, w) + 1) / 2
	b = None
	if bias is None: b = 1.
	if bias == 'nc': b = nc(w)
	if isinstance(bias, torch.Tensor): b = bias.view(1, -1, 1, 1)
	if b is None: raise ValueError("Invalid value for parameter bias: " + str(bias))
	return s.pow(b)
# Corresponding normalization condition functions for heuristic variance determination
def nc_raised_ang_sim(w):
	return 1/(-math.log( 1 - 1 / (w.size(0) ** (1 / (utils.shape2size(tuple(w[0].size()))))) ))

# Get exponential similarity function e^(-f^p/var)
def get_exp_sim(f, nc=None):
	def exp_sim(x, w, bias=None):
		d = f(x, w)
		b = None
		if bias is None: b = 1.
		if bias == 'nc':
			if nc is None: raise ValueError("Normalization condition was invoked but no corresponding function was provided")
			b = nc(w)
		if isinstance(bias, torch.Tensor): b = bias.view(1, -1, 1, 1)
		if b is None: raise ValueError("Invalid value for parameter bias: " + str(bias))
		return torch.exp(-d * b)
	return exp_sim
# Distance functions for the exponential similarity
def dist_euclid(x, w): # Euclidean distance for gauss-rbf similarity
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	return dist(x_unf.view(-1, x.size(4)), w.view(w.size(0), -1))
def dist_cos(x, w): # Cosine distance, which is equivalent to Euclidean distance between normalized vectors
	cos_sim = cos_sim2d(x, w)
	return (1 - cos_sim)/2
def dist_ang(x, w): # Angular distance
	cos_sim = cos_sim2d(x, w)
	return torch.acos(cos_sim)/math.pi
# Normalization condition functions for heuristic variance determination
def nc_base(w):
	return w.size(0) ** (1 / (utils.shape2size(tuple(w[0].size()))))
def nc_max_dist(w):
	return nc_base(w) / dist(w.view(w.size(0), -1), w.view(w.size(0), -1)).max().detach().item()
def nc_cos_dist(w):
	return 2 / (1 - torch.cos(math.pi / nc_base(w)).detach().item())
def nc_ang_dist(w):
	return nc_base(w)

# Get affine version of a similarity function
def get_affine_sim(sim, b=0., s=1., p=1.):
	def biased_sim(x, w, bias=None):
		return sim(x, w, bias).pow(p) * s + b
	return biased_sim

# Get power version of a normcond function
def get_pow_nc(nc, p=1.):
	def pow_nc(w):
		return nc(w) ** p
	return pow_nc

# The identity function
def identity(x):
	return x

# Step nonlinearity
def step(x):
	return (x >= 0).float()

# Relu nonlinearity
def relu(x):
	return x.clamp(0)

# Clamping nonlinearity
def clamp(x):
	return x.clamp(-1, 1)

# Shrinkage nonlinearity
def shrink(x):
	return x - clamp(x)

# Tanh nonlinearity
def tanh(x):
	return x.tanh()

# Soft shrink nonlinearity
def sshrink(x):
	return x - tanh(x)

# Rectified clamp nonlinearity
def reclamp(x):
	return clamp(x) * step(x)

# Rectified shrink nonlinearity
def reshrink(x):
	return shrink(x) * step(x)

# Rectified tanh nonlinearity
def retanh(x):
	return tanh(x) * step(x)

# Rectified shoft shrink nonlinearity
def resshrink(x):
	return sshrink(x) * step(x)

# Mixed shrink-clamp nonlinearity
def shramp(x):
	return shrink(x) * (1 - step(x)) + clamp(x) * step(x)

# Mixed sshrink-tanh nonlinearity
def sshranh(x):
	return sshrink(x) * (1 - step(x)) + tanh(x) * step(x)

# Gauss nonlinearity
def gauss(x):
	return torch.exp(-(x**2) / 2)

# Get affine version of a nonlinearity
def get_affine_act(nonlin, scale_in=1., scale_out=1., offset_in=0., offset_out=0., p=1):
	def affine_act(x):
		return scale_out * nonlin(x / scale_in + offset_in).pow(p) + offset_out
	return affine_act


# Competitive nonlinearity: k-WTA
def kwta(x, k=1, t=None):
	if t is not None: x = x * t # Teacher signal is used to drive competition, if provided
	#return (x == x.max(1, keepdim=True)[0]).float() # WTA
	return (x >= x.kthvalue(x.size(1) - k + 1, dim=1, keepdim=True)[0]).float() # k-WTA

# Competitive nonlinearity: shifted soft-WTA
def ssoftwta(x, k=0, t=None):
	x = x + k
	if t is not None: x = x * t # Teacher signal is used to drive competition, if provided
	x_sum = x.sum(dim=1, keepdim=True)
	x_sum = x_sum + (x_sum == 0).float() # Prevent divisions by 0
	return x/x_sum

# Competitive nonlinearity: polynomial soft-WTA
def psoftwta(x, k=1, t=None):
	return ssoftwta(x.pow(1/k), 0, t)

# Competitive nonlinearity: exponential soft-WTA
def esoftwta(x, k=1, t=None):
	return ssoftwta(torch.exp(x/k), 0, t)

# Competitive nonlinearity: threshold
def threshcomp(x, k=0, t=None):
	if t is not None: x = x * t # Teacher signal is used to drive competition, if provided
	return (x + k > 0).float()


# A modified batchnorm layer that preserves per-feature relative variance information. This is done by multiplying the
# output of an ordinary bn layer by a coefficient given by the ratio between the running variance and the aggregated
# (average) running variances, so that variables will preserve their relative variance, while remaining normalized
# (unit variance) on average.
def modified_bn(bn_layer, input):
	shape = (1, -1, *(1 for _ in range(len(input.size()) - 2)))
	out = bn_layer(input)
	out *= ((bn_layer.running_var.view(*shape) + bn_layer.eps)/(bn_layer.running_var.mean() + bn_layer.eps))**0.5
	return out


# A custom module that takes a constant and transforms it into a parameter that is function of other parameters,
# thus giving a gradient correction in the backward pass
class GradCorrector(nn.Module):
	def __init__(self):
		super(GradCorrector, self).__init__()
		self.Fn = None
	
	def forward(self, x, p):
		return p

# A gradient corrector for a mean parameter
class MeanGradCorrector(GradCorrector):
	def __init__(self, beta=0.1):
		super().__init__()
		
		class Fn(torch.autograd.Function):
			@staticmethod
			def forward(ctx, x, p):
				ctx.save_for_backward(x, p)
				return p
			
			@staticmethod
			def backward(ctx, e):
				x, p = ctx.saved_tensors
				N = x.size(0) * x.size(2)
				self.track(x, p, e)
				grad_x = self.running_e.view(1, -1, 1) * torch.ones_like(x).float() / N
				return grad_x, e
		
		self.Fn = Fn
		
		self.register_buffer('running_e', torch.tensor(0.))
		
		self.beta = beta
	
	def track(self, x, p, e):
		self.running_e = self.running_e + self.beta * (e - self.running_e)
		
	def forward(self, x, p):
		return self.Fn.apply(x, p)


# A gradient corrector for a standard deviation parameter
class StdGradCorrector(GradCorrector):
	def __init__(self, beta=0.1):
		super().__init__()
		
		class Fn(torch.autograd.Function):
			@staticmethod
			def forward(ctx, x, p):
				ctx.save_for_backward(x, p)
				return p
			
			@staticmethod
			def backward(ctx, e):
				x, p = ctx.saved_tensors
				N = x.size(0) * x.size(2)
				self.track(x, p, e)
				grad_x = self.running_e.view(1, -1, 1) * (x - self.running_x.view(1, -1, 1)) / (p * N)
				return grad_x, e
		
		self.Fn = Fn
		
		self.register_buffer('running_x', torch.tensor(0.))
		self.register_buffer('running_e', torch.tensor(0.))
		
		self.beta = beta
	
	def track(self, x, p, e):
		mean = x.mean(dim=(0, 2))
		self.running_x = self.running_x + self.beta * (mean - self.running_x)
		self.running_e = self.running_e + self.beta * (e - self.running_e)
		
	def forward(self, x, p):
		return self.Fn.apply(x, p)


