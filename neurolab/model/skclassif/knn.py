import torch
from sklearn.neighbors import KNeighborsClassifier

from ... import params as P
from .skclassif import SkClassif


class KNNClassifier(SkClassif):
	
	def __init__(self, config, input_shape=None):
		super(KNNClassifier, self).__init__(config, input_shape)
		
		self.N_NEIGHBORS = config.CONFIG_OPTIONS.get(P.KEY_KNN_N_NEIGHBORS, 10)
		self.clf = KNeighborsClassifier(n_neighbors=self.N_NEIGHBORS)
	
	def get_clf_pred(self, x):
		if not self.clf_fitted: return torch.rand((len(x), self.NUM_CLASSES), device=P.DEVICE)
		return torch.tensor(self.clf.predict_proba(self.nystroem.transform(x)), device=P.DEVICE)
	
	