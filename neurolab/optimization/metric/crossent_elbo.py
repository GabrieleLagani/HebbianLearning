from ..optimization import MetricManager
from neurolab import params as P
from . import ELBOMetric
from . import CrossEntMetric


class CrossEntELBOMetric:
	def __init__(self, alpha, beta):
		self.ALPHA = alpha
		self.BETA = beta
		self.elbo_loss = ELBOMetric(self.BETA)
		self.crossent_loss = CrossEntMetric()
	
	def __call__(self, outputs, targets):
		return ((self.ALPHA * self.elbo_loss(outputs, targets)) if self.ALPHA != 0 else 0.) \
		       + (((1 - self.ALPHA) * self.crossent_loss(outputs, targets)) if self.ALPHA != 1 else 0.)

# Criterion manager for ELBO
class CrossEntELBOMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
		self.ALPHA = config.CONFIG_OPTIONS.get(P.KEY_ALPHA, 1.)
		self.BETA = config.CONFIG_OPTIONS.get(P.KEY_ELBO_BETA, 1.)
	
	def get_metric(self):
		return CrossEntELBOMetric(self.ALPHA, self.BETA)
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "crossent-elbo"

