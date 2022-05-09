import torch.optim.lr_scheduler as sched
from collections import Counter

from ... import params as P
from ..optimization import SchedManager
from neurolab import utils


# Scheduling manager for multistep lr scheduling
class MultiStepSchedManager(SchedManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_scheduler(self, optimizer, saved_state=None):
		gamma = self.config.CONFIG_OPTIONS[P.KEY_LR_DECAY]
		milestones = self.config.CONFIG_OPTIONS[P.KEY_MILESTONES]
		sch = sched.MultiStepLR(optimizer, gamma=gamma, milestones=milestones)
		if saved_state is not None:
			sch = utils.dict2obj(saved_state, sch) # The load_state_dict method of scheduler restores all the information, including initialization parameters
			# Here, we reset init parameters to the value provided in the config, which might be different from the
			# old saved value, because one might want to change the scheduling parameters and then resume training with
			# the new parameters. As an alternative, one could redefine the load_state_dict method, but that would be
			# more complex.
			sch.gamma = gamma
			sch.milestones = Counter(milestones)
			# The following is another way by which we should be able to recover the saved state while keeping the new
			# initialization parameters, but the solution above is more general.
			#sch = sched.MultiStepLR(optimizer, gamma=gamma, milestones=milestones, last_epoch=saved_state['last_epoch'])
		return sch
