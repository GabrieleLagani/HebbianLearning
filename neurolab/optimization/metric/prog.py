import torch

from ..optimization import MetricManager
from neurolab import params as P


# Return index representing current training progress (number of processed batches). Useful when one does not want to
# use early stopping
class ProgMetric:
	def __init__(self):
		self.idx = 0
	
	def __call__(self, outputs, targets):
		self.idx += 1
		return self.idx

# Criterion manager for accuracy
class ProgMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return ProgMetric()
	
	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "progress"