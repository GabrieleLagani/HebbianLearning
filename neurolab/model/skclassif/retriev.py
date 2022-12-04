import torch
from sklearn.neighbors import KNeighborsClassifier

from ... import params as P
from .skclassif import SkClassif


class Retriever(SkClassif):
	
	RETR_CLASSES = 'retr_classes'
	
	def __init__(self, config, input_shape=None):
		super(Retriever, self).__init__(config, input_shape)
		
		self.N_NEIGHBORS = config.CONFIG_OPTIONS.get(P.KEY_KNN_N_NEIGHBORS, max(list(config.CONFIG_OPTIONS.get(P.KEY_RETR_K, 10))))
		self.clf = KNeighborsClassifier(n_neighbors=self.N_NEIGHBORS)
	
	def compute_output(self, x):
		out = {}
		
		clf_out = self.get_clf_pred(x)
		
		out[self.CLF] = clf_out
		out[self.RETR_CLASSES] = {P.KEY_RETR_CLASSES: clf_out}
		
		return out
	
	def get_clf_pred(self, x):
		if not self.clf_fitted: return torch.randint(0, self.NUM_CLASSES, (len(x), self.N_NEIGHBORS), device=P.DEVICE)
		_, idx = self.clf.kneighbors(x)
		return torch.tensor(self.clf._y[idx], device=P.DEVICE)
	
	