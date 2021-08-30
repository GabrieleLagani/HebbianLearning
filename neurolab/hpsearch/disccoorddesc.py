import random

from .hpsearch import HPManager
from .. import utils
from .. import params as P

# Hyperparam search based on coordinate descent on discretized space
class DiscCoordDescHPManager(HPManager):
	def __init__(self, config_options, hpsearch_num, hpseed):
		super().__init__(config_options, hpsearch_num, hpseed)
		
		self.hyperparams = config_options[P.KEY_HPSEARCH_PARAMS]
		
		self.curr_key = -1
		self.curr_direction = 1
		self.base_coords = None
		self.curr_coords = None
		self.curr_hyperparams = None
		self.best_result = None
		self.best_hyperparams = None
		self.best_coords = None
		self.last_modified = 0
		self.last_modified_direction = 0
	
	def state_dict(self):
		return {
			'curr_key': self.curr_key,
			'curr_direction': self.curr_direction,
			'base_coords': self.base_coords,
			'curr_coords': self.curr_coords,
			'curr_hyperparams': self.curr_hyperparams,
			'best_result': self.best_result,
			'best_hyperparams': self.best_hyperparams,
			'best_coords': self.best_coords,
			'last_modified': self.last_modified,
			'last_modified_direction': self.last_modified_direction,
			'rng': self.rng_state,
		}
	
	def load_state_dict(self, d):
		self.curr_key = d['curr_key']
		self.curr_direction = d['curr_direction']
		self.base_coords = d['base_coords']
		self.curr_coords = d['curr_coords']
		self.curr_hyperparams = d['curr_hyperparams']
		self.best_result = d['best_result']
		self.best_hyperparams = d['best_hyperparams']
		self.best_coords = d['best_coords']
		self.last_modified = d['last_modified']
		self.last_modified_direction = d['last_modified_direction']
		self.rng_state = d['rng']
	
	def get_next_hyperparams(self):
		if self.curr_coords == None: self.curr_coords = {k: random.randint(0, len(self.hyperparams[k]) - 1) for k in self.hyperparams.keys()}
		else:
			self.curr_direction = 1 if self.curr_direction == -1 else -1
			if self.curr_direction == -1:
				self.curr_key = (self.curr_key + 1) % len(self.hyperparams.keys()) # Handle borders like a torus
				self.last_modified += 1
				if self.last_modified == len(self.hyperparams.keys()) + 1: return None
				self.base_coords = self.best_coords.copy()
			k = list(self.hyperparams.keys())[self.curr_key]
			# Check if we already tried this configuration and in case skip to the next one
			if (len(self.hyperparams[k]) <= 1) \
					or (len(self.hyperparams[k]) == 2 and self.curr_direction == 1) \
					or (self.last_modified == len(self.hyperparams.keys()) and self.curr_direction != self.last_modified_direction) \
					or (self.last_modified == len(self.hyperparams.keys()) and len(self.hyperparams[k]) <= 3):
				return next(self)
			self.curr_coords = self.base_coords.copy()
			self.curr_coords[k] = (self.base_coords[k] + self.curr_direction) % len(self.hyperparams[k])
		
		# Return the specific hyperparameters
		self.curr_hyperparams = {k: self.hyperparams[k][self.curr_coords[k]] for k in self.hyperparams.keys()}
		return self.curr_hyperparams
	
	def update(self, result):
		if utils.is_better(result, self.best_result, P.HIGHER_IS_BETTER):
			self.best_result = result
			self.best_hyperparams = self.curr_hyperparams
			self.best_coords = self.curr_coords.copy()
			self.last_modified = 0
			self.last_modified_direction = self.curr_direction

