from ..optimization import MetricManager
from neurolab import params as P
from . import ELBOMetric
from . import CrossEntMetric


class CrossEntELBOMetric:
	def __init__(self, alpha_s, alpha_u, beta):
		self.ALPHA_S = alpha_s
		self.ALPHA_U = alpha_u
		self.BETA = beta
		self.elbo_loss = ELBOMetric(self.BETA)
		self.crossent_loss = CrossEntMetric()
	
	def __call__(self, outputs, targets):
		return self.ALPHA_S * self.crossent_loss(outputs, targets) + self.ALPHA_U * self.elbo_loss(outputs, targets)

# Criterion manager for ELBO
class CrossEntELBOMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
		self.ALPHA_S = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_S, 1.)
		self.ALPHA_U = config.CONFIG_OPTIONS.get(P.KEY_ALPHA_U, 1.)
		self.BETA = config.CONFIG_OPTIONS.get(P.KEY_ELBO_BETA, 1.)
	
	def get_metric(self):
		return CrossEntELBOMetric(self.ALPHA_S, self.ALPHA_U, self.BETA)
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "crossent-elbo"

