import torch.optim as optim

from ... import params as P
from ..optimization import OptimManager


# Optimization manager for SGD
class SGDOptimManager(OptimManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_optimizer(self, param_groups):
		opt = optim.SGD(param_groups, lr=self.config.CONFIG_OPTIONS.get(P.KEY_LEARNING_RATE, 1e-3),
		                 momentum=self.config.CONFIG_OPTIONS.get(P.KEY_MOMENTUM, 0.),
		                 weight_decay=self.config.CONFIG_OPTIONS.get(P.KEY_L2_PENALTY, 0.),
		                 nesterov=self.config.CONFIG_OPTIONS.get(P.KEY_NESTEROV, self.config.CONFIG_OPTIONS.get(P.KEY_MOMENTUM, 0.) > 0.))
		return opt