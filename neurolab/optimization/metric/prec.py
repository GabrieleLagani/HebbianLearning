import torch

from ..optimization import MetricManager
from neurolab import params as P


# Evaluate outputs on a retrieval experiment basis: outputs must contain list of retrieved elements classes for each
# query/input. Return precision score @k (number of retrieved elements), averaged over all the queries in the batch.
class PrecMetric:
	def __init__(self, k):
		self.K = k
		
	def __call__(self, outputs, targets):
		num_retr = outputs.size(1)
		k = min(num_retr, self.K)
		outputs = (outputs == targets.view(-1, 1)).float()
		prec_at_k = outputs[:, :k].sum(dim=1)/k
		return prec_at_k.mean().item()

# Criterion manager for precision score @k
class PrecMetricManager(MetricManager):
	# The default values to use for k. Use just k=10 by default. The user might require to evaluate different precision
	# scores with different k. We use these static variables below to keep track of the k value to be used to
	# instantiate the next PrecMetricManager.
	k_list = [10]
	k_list_index = 0
	
	def __init__(self, config):
		super().__init__(config)
		
		# Retrieve the list of k from config
		PrecMetricManager.k_list = config.CONFIG_OPTIONS.get(P.KEY_RETR_K, PrecMetricManager.k_list)
		if type(PrecMetricManager.k_list) is not list: PrecMetricManager.k_list = list(PrecMetricManager.k_list)
		if len(PrecMetricManager.k_list) != (
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.PrecMetricManager') +
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.prec.PrecMetricManager')):
			raise ValueError("The number of K values for precision score must match the number of precision score metric managers provided for criterion evaluation")
		
		# Determine the value of k to be used for the precision score evaluation
		self.K = PrecMetricManager.k_list[PrecMetricManager.k_list_index]
		PrecMetricManager.k_list_index = (PrecMetricManager.k_list_index + 1) % len(PrecMetricManager.k_list)
	
	def get_metric(self):
		return PrecMetric(self.K)

	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "precision-at" + str(self.K)