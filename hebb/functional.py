import torch
import torch.nn.functional as F

import params as P
from neurolab import utils


# Apply unfold operation to x in order to prepare it to be processed against a sliding kernel whose shape
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

# Custom vectorial function representing sum of an x with a sliding kernel, just like convolution is multiplication
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


# A modified batchnorm layer that preserves per-feature relative variance information. This is done by multiplying the
# output of an ordinary bn layer by a coefficient given by the ratio between the running variance and the aggregated
# (average) running variances, so that variables will preserve their relative variance, while remaining normalized
# (unit variance) on average.
def modified_bn(bn_layer, input):
	shape = (1, -1, *(1 for _ in range(len(input.size()) - 2)))
	out = bn_layer(input)
	out *= ((bn_layer.running_var.view(*shape) + bn_layer.eps)/(bn_layer.running_var.mean() + bn_layer.eps))**0.5
	return out


# The identity function
def identity(x):
	return x

# Compute product between x and sliding kernel
def kernel_mult2d(x, w, b=None):
	return F.conv2d(x, w, b)

# Projection of x on weight vectors
def vector_proj2d(x, w, bias=None):
	# Compute scalar product with sliding kernel
	prod = kernel_mult2d(x, w)
	# Divide by the norm of the weight vector to obtain the projection
	norm_w = torch.norm(w.view(w.size(0), -1), p=2, dim=1).view(1, -1, 1, 1)
	norm_w = norm_w + (norm_w == 0).float()  # Prevent divisions by zero
	if bias is None: return prod / norm_w
	return prod / norm_w + bias.view(1, -1, 1, 1)

# Cosine similarity between an x map and a sliding kernel
def cos_sim2d(x, w, bias=None):
	proj = vector_proj2d(x, w)
	# Divide by the norm of the x to obtain the cosine similarity
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	norm_x = torch.norm(x_unf, p=2, dim=4)
	norm_x += (norm_x == 0).float()  # Prevent divisions by zero
	if bias is None: return proj / norm_x
	return (proj / norm_x + bias.view(1, -1, 1, 1)).clamp(-1, 1)

# Cosine similarity remapped to 0, 1
def raised_cos2d(x, w, bias=None):
	return (cos_sim2d(x, w, bias) + 1) / 2

# Returns function that computes raised cosine power p
def raised_cos2d_pow(p=2):
	def raised_cos2d_pow_p(x, w, bias=None):
		if bias is None: return raised_cos2d(x, w).pow(p)
		return (raised_cos2d(x, w).pow(p) + bias.view(1, -1, 1, 1)).clamp(0, 1)
	
	return raised_cos2d_pow_p

# Response of a gaussian activation function
VAR_HEUR_NUM_DIMS = 'num_dims'
VAR_HEUR_NORM_COND = 'norm_cond'
VAR_HEUR_MEAN_DIST = 'mean_dist'
def gauss(x, w, var=None): # var=VAR_HEUR_NUM_DIMS):
	# Serialized version of distance computation using less memory
	x_unf = unfold_map2d(x, w.size(2), w.size(3))
	d = torch.zeros(x_unf.size(0), w.size(0), x_unf.size(2), x_unf.size(3),
	                device=x.device)  # batch, out-ch, height, width
	for i in range(w.size(0) // P.HEBB_UPD_GRP + (1 if w.size(0) % P.HEBB_UPD_GRP != 0 else 0)):
		start = i * P.HEBB_UPD_GRP
		end = min((i + 1) * P.HEBB_UPD_GRP, w.size(0))
		w_i = w[start:end, :, :, :]
		w_i = w_i.view(1, w_i.size(0), 1, 1, -1)  # batch, out-ch, height, width, filter
		d[:, start:end, :, :] = torch.norm(x_unf - w_i, p=2, dim=4)  # w_i broadcast over x_unf batch, height and width dims, x_unf broadcast over w out_ch dim
	if var == VAR_HEUR_NUM_DIMS: return torch.exp(-d.pow(2) / (2 * utils.shape2size(tuple(w[0].size()))))  # heuristic: use number of dimensions as variance
	if var == VAR_HEUR_NORM_COND: return torch.exp(-d.pow(2) / (2 * torch.norm(w.view(w.size(0), 1, -1) - w.view(1, w.size(0), -1), p=2, dim=2).max().pow(2)/(w.size(0)**(2/(utils.shape2size(tuple(w[0].size()))))))) # heuristic: normalization condition
	if var == VAR_HEUR_MEAN_DIST: return torch.exp(-d.pow(2) / (2 * d.mean().pow(2))) # heuristic: use mean distance as variance
	if var is None: return torch.exp(-d.pow(2) / 2)
	return torch.exp(-d.pow(2) / (2 * var.view(1, -1, 1, 1)))


# Clamping nonlinearity
def clamp(x):
	return x.clamp(-1, 1)

# Shrinkage nonlinearity
def shrink(x):
	return x - x.clamp(-1, 1)

# Tanh nonlinearity
def tanh(x):
	return x.tanh()

# Soft shrink nonlinearity
def sshrink(x):
	return x - x.tanh()


# Competitive nonlinearity: k-WTA
def kwta(x, k=1):
	#return (x == x.max(1, keepdim=True)[0]).float() # WTA
	return (x >= x.kthvalue(x.size(1) - k + 1, dim=1, keepdim=True)[0]).float() # k-WTA

# Competitive nonlinearity: soft-WTA
def esoftwta(x, k=1):
	return torch.softmax(x/k, dim=1)

# Competitive nonlinearity: polynomial soft-WTA
def psoftwta(x, k=1):
	norm_x = torch.norm(x, p=1/k, dim=1, keepdim=True)
	norm_x = norm_x + (norm_x == 0).float()  # Prevent divisions by zero
	return (x/norm_x).pow(1/k)

