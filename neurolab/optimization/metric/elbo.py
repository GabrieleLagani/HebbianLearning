import torch
import torch.nn.functional as F

from ..optimization import MetricManager
from neurolab import params as P


class ELBOMetric:
	def __init__(self, beta):
		self.BETA = beta
	
	def __call__(self, outputs, targets):
		if isinstance(targets, dict): targets = targets[P.KEY_RECONSTR_TARGETS]
		reconstr = outputs[P.KEY_AUTOENC_RECONSTR]
		mu = outputs[P.KEY_ELBO_MU]
		log_var = outputs[P.KEY_ELBO_LOG_VAR]
		reconstr_loss = F.mse_loss(reconstr, targets)
		kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
		return reconstr_loss + self.BETA * kld_loss

# Criterion manager for ELBO
class ELBOMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
		self.BETA = config.CONFIG_OPTIONS.get(P.KEY_ELBO_BETA, 1.)
	
	def get_metric(self):
		return ELBOMetric(self.BETA)
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "elbo"

