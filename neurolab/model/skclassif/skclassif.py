import torch
from sklearn.kernel_approximation import Nystroem

from ... import params as P
from ... import utils
from ..model import Model

# Base class as wrapper for classifiers from scikit learn
class SkClassif(Model):
	# Layer names
	CLF = 'clf'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(SkClassif, self).__init__(config, input_shape)
		
		self.INPUT_SIZE = utils.shape2size(self.get_input_shape())
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		
		self.NUM_SAMPLES = config.CONFIG_OPTIONS.get(P.KEY_SKCLF_NUM_SAMPLES, config.CONFIG_OPTIONS.get(P.KEY_NUM_TRN_SAMPLES, config.CONFIG_OPTIONS.get(P.KEY_TOT_TRN_SAMPLES, P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_TRN_SET_SIZE])))
		self.N_COMPONENTS = config.CONFIG_OPTIONS.get(P.KEY_NYSTROEM_N_COMPONENTS, 0)
		if self.N_COMPONENTS is None: self.N_COMPONENTS = 0
		self.N_COMPONENTS = min(self.N_COMPONENTS, self.NUM_SAMPLES)
		self.nystroem = Nystroem(n_components=self.N_COMPONENTS) if self.N_COMPONENTS > 0 else None
		self.clf = None
		self.nystroem_fitted = False
		self.clf_fitted = False
		self.X = []
		self.X_transformed = []
		self.y = []
		self.binarize = config.CONFIG_OPTIONS.get(P.KEY_SKCLF_BINARIZE, True)
		self.normalize = config.CONFIG_OPTIONS.get(P.KEY_SKCLF_NORM, False)
	
	def state_dict(self):
		d = super(SkClassif, self).state_dict()
		d['nystroem'] = self.nystroem
		d['clf'] = self.clf
		d['nystroem_fitted'] = self.nystroem_fitted
		d['clf_fitted'] = self.clf_fitted
		return d
	
	def load_state_dict(self, state_dict, strict = ...):
		self.nystroem = state_dict.pop('nystroem')
		self.clf = state_dict.pop('clf')
		self.nystroem_fitted = state_dict.pop('nystroem_fitted')
		self.clf_fitted = state_dict.pop('clf_fitted')
		super(SkClassif, self).load_state_dict(state_dict, strict)
	
	def norm_if_needed(self, x):
		if not self.normalize: return x
		norm_x = x.norm(p=2, dim=1, keepdim=True)
		norm_x += (norm_x == 0).float()  # Prevent divisions by zero
		return x / norm_x
	
	def binarize_if_needed(self, x):
		if not self.binarize: return x
		return (x >= 0).float()
	
	# Here we define the flow of information through the network
	def forward(self, x):
		x = self.binarize_if_needed(x.view(x.size(0), -1)).tolist() # Binarize input if needed
		
		# Here we append inputs to training pipeline if we are in training mode
		if self.training:
			if not self.clf_fitted:
				# Here we use just the first NUM_SAMPLES samples to do a Nystroem approximation, because they are already
				# a random subset of the dataset. This allows to save memory by avoiding to store the whole dataset.
				if self.nystroem is not None:
					if not self.nystroem_fitted:
						self.X += x
						if len(self.X) >= self.N_COMPONENTS:
							self.X_transformed = self.norm_if_needed(torch.tensor(self.nystroem.fit_transform(self.X), device=P.DEVICE)).tolist()
							self.nystroem_fitted = True
							self.X = []
					else: self.X_transformed += self.norm_if_needed(torch.tensor(self.nystroem.transform(x), device=P.DEVICE)).tolist()
				else: self.X_transformed += self.norm_if_needed(torch.tensor(x, device=P.DEVICE)).tolist()
				
				# Here we fit the actual classifier
				if len(self.X_transformed) >= self.NUM_SAMPLES:
					self.clf.fit(self.X_transformed, self.y)
					self.clf_fitted = True
					self.X_transformed = []
					self.y = []
		
		return self.compute_output(self.norm_if_needed(self.nystroem.transform(x) if self.nystroem is not None else x))
	
	# Process incput batch and compute output dictionary
	def compute_output(self, x):
		out = {}
		
		clf_out = self.get_clf_pred(x)
		
		out[self.CLF] = clf_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: clf_out}
		
		return out
	
	# Returns classifier predictions for a given input batch
	def get_clf_pred(self, x):
		if not self.clf_fitted: return torch.rand((len(x), self.NUM_CLASSES), device=P.DEVICE)
		return utils.dense2onehot(torch.tensor(self.clf.predict(x), device=P.DEVICE), self.NUM_CLASSES)
		
	# Set label info for current batch
	def set_teacher_signal(self, y):
		if y is not None and len(self.y) < self.NUM_SAMPLES and self.training and not self.clf_fitted: self.y += y.tolist()
