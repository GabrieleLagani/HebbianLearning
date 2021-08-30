import torch

from ... import params as P
from ..optimization import MetricManager


# Return number of correctly classified samples given network outputs and target labels, counting a hit if the correct
# label is in the top k outputs. The experiment class is then going to average the result over a batch of data, thus
# finally obtaining an accuracy score.
class TopKAccMetric:
	def __init__(self, k):
		self.K = k
	
	def __call__(self, outputs, targets):
		if isinstance(outputs, dict): outputs = outputs[P.KEY_CLASS_SCORES]
		if isinstance(targets, dict): targets = targets[P.KEY_LABEL_TARGETS]
		_, pred = torch.topk(outputs, self.K)
		return (pred == targets.view(-1, 1)).float().sum(dim=1).mean().item()

# Criterion manager for top-k accuracy.
class TopKAccMetricManager(MetricManager):
	# The default values to use for k. Use just k=1 by default. The user might require to evaluate differet top-
	# k accuracies with different k. We use these static variables above to keep track of the k value to be used to
	# instantiate the next TopKAccMetricManager.
	k_list = [1]
	k_list_index = 0
	
	def __init__(self, config):
		super().__init__(config)
		
		# Retrieve the list of k from config
		TopKAccMetricManager.k_list = config.CONFIG_OPTIONS.get(P.KEY_TOPKACC_K, TopKAccMetricManager.k_list)
		if type(TopKAccMetricManager.k_list) is not list: TopKAccMetricManager.k_list = list(TopKAccMetricManager.k_list)
		if len(TopKAccMetricManager.k_list) != (
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.TopKAccMetricManager') +
				config.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER].count('neurolab.optimization.metric.topkacc.TopKAccMetricManager')):
			raise ValueError("The number of K values for top-k accuracy must match the number of top-k accuracy metric managers provided for criterion evaluation")
		
		# Determine the value of k to be used for the top-k accuracy evaluation
		self.K = TopKAccMetricManager.k_list[TopKAccMetricManager.k_list_index]
		TopKAccMetricManager.k_list_index = (TopKAccMetricManager.k_list_index + 1) % len(TopKAccMetricManager.k_list)
	
	def get_metric(self):
		return TopKAccMetric(self.K)
	
	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "top" + str(self.K) + "-accuracy"