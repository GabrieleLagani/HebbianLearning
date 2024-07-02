import torch

# Computes shape of output tensor of a model for a given input tensor shape
def get_output_fmap_shape(model, input_shape):
	training = model.training
	model.eval()
	with torch.no_grad(): # We are disabling gradient computation for processing simulated inputs
		# Generate simulated x, feed to the network, and get corresponding output
		dummy_input = torch.ones(1, *input_shape, requires_grad=False)
		out = model(dummy_input)
		output_shape = tens2shape(out)
	model.train(training)
	return output_shape

# Convert tensor shape to total tensor size
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Convert (dictionary of) tensors to (dictionary of) corresponding shapes
def tens2shape(input):
	return {k: tens2shape(input[k]) for k in input} if isinstance(input, dict) else tuple(input.size())[1:]

# Convert dense-encoded vector to one-hot encoded
def dense2onehot(tensor, n):
	return torch.zeros(tensor.size(0), n, device=tensor.device).scatter_(1, tensor.unsqueeze(1).long(), 1)

