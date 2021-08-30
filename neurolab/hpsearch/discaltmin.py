from .hpsearch import HPManager
from .. import utils
from .. import params as P

# Hyperparam search based on alternating minimization on discretized space
class DiscAltMinHPManager(HPManager):
	def __init__(self, config_options, hpsearch_num, hpseed):
		super().__init__(config_options, hpsearch_num, hpseed)
		
		self.hyperparams = config_options[P.KEY_HPSEARCH_PARAMS]
		
		self.curr_key = 0
		self.curr_index = -1
		self.curr_hyperparams = None
		self.best_result = None
		self.best_hyperparams = None
		self.last_modified = 0
	
	def state_dict(self):
		return {
			'curr_key': self.curr_key,
			'curr_index': self.curr_index,
			'curr_hyperparams': self.curr_hyperparams,
			'best_result': self.best_result,
			'best_hyperparams': self.best_hyperparams,
			'last_modified': self.last_modified,
		}
	
	def load_state_dict(self, d):
		self.curr_key = d['curr_key']
		self.curr_index = d['curr_index']
		self.curr_hyperparams = d['curr_hyperparams']
		self.best_result = d['best_result']
		self.best_hyperparams = d['best_hyperparams']
		self.last_modified = d['last_modified']
	
	def get_next_hyperparams(self):
		self.curr_index += 1
		if self.curr_index >= len(self.hyperparams[list(self.hyperparams.keys())[self.curr_key]]):
			self.curr_index = 0
			self.curr_key = (self.curr_key + 1) % len(self.hyperparams.keys())
			self.last_modified += 1
			if self.last_modified == len(self.hyperparams.keys()): return None
		self.curr_hyperparams = self.best_hyperparams.copy() if self.best_hyperparams is not None else {k: self.hyperparams[k][0] for k in self.hyperparams.keys()}
		k = list(self.hyperparams.keys())[self.curr_key]
		self.curr_hyperparams[k] = self.hyperparams[k][self.curr_index]
		# Check if we already tried this configuration and in this case skip to the next one
		if self.best_hyperparams is not None and self.curr_hyperparams[k] == self.best_hyperparams[k]: return next(self)
		# Return the specific hyperparameters
		return self.curr_hyperparams
	
	def update(self, result):
		if utils.is_better(result, self.best_result, P.HIGHER_IS_BETTER):
			self.best_result = result
			self.best_hyperparams = self.curr_hyperparams
			self.last_modified = 0

