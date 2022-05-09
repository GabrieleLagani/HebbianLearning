import torch

from ..optimization import MetricManager
from neurolab import params as P


# Evaluate outputs on a retrieval experiment basis: outputs must contain list of retrieved elements classes for each
# query/input. Return average precision score @k (number of retrieved elements), averaged over all the queries in the
# batch (Mean Average Precision - MAP).
class MAPMetric:
	def __init__(self, k, num_rel):
		self.K = k
		self.NUM_REL = num_rel
		
	def __call__(self, outputs, targets):
		num_retr = outputs.size(1)
		k = min(num_retr, self.K)
		outputs = (outputs == targets.view(-1, 1)).float()
		prec_at_i = outputs.cumsum(dim=1)/torch.arange(1, num_retr + 1, device=P.DEVICE).float().view(1, -1)
		map = (outputs[:, :k] * prec_at_i[:, :k]).sum(dim=1)/min(k, self.NUM_REL)
		return map.mean().item()

# Criterion manager for average precision score @k
class MAPMetricManager(MetricManager):
	# The default values to use for k. Use just k=10 by default. The user might require to evaluate different precision
	# scores with different k. We use these static variables below to keep track of the k value to be used to
	# instantiate the next MAPMetricManager.
	k_list = [10]
	k_list_index = 0
	
	def __init__(self, config):
		super().__init__(config)
		
		# Retrieve the list of k from config
		MAPMetricManager.k_list = config.CONFIG_OPTIONS.get(P.KEY_RETR_K, MAPMetricManager.k_list)
		if type(MAPMetricManager.k_list) is not list: MAPMetricManager.k_list = list(MAPMetricManager.k_list)
		if len(MAPMetricManager.k_list) != (
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.MAPMetricManager') +
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.map.MAPMetricManager')):
			raise ValueError("The number of K values for precision score must match the number of MAP metric managers provided for criterion evaluation")
		
		# Determine the value of k to be used for the precision score evaluation
		self.K = MAPMetricManager.k_list[MAPMetricManager.k_list_index]
		MAPMetricManager.k_list_index = (MAPMetricManager.k_list_index + 1) % len(MAPMetricManager.k_list)
		
		# Other parameters
		self.NUM_REL = config.CONFIG_OPTIONS.get(P.KEY_RETR_NUM_REL, self.K) # Number of samples per query considered relevant in map computation
	
	def get_metric(self):
		return MAPMetric(self.K, self.NUM_REL)

	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "map-at" + str(self.K)