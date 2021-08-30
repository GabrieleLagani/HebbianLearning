import torch.optim.lr_scheduler as sched

from ... import params as P
from ..optimization import SchedManager


# Scheduling manager for multistep lr scheduling
class MultiStepSchedManager(SchedManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_scheduler(self, optimizer):
		return sched.MultiStepLR(optimizer, gamma=self.config.CONFIG_OPTIONS[P.KEY_LR_DECAY], milestones=self.config.CONFIG_OPTIONS[P.KEY_MILESTONES])
