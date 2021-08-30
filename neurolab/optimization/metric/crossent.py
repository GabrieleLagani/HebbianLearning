import torch.nn as nn

from ..optimization import MetricManager
from neurolab import params as P


# Wrapper around Pytorch CrossEntropyLoss criterion
class CrossEntMetric:
	def __init__(self):
		self.crossent_loss = nn.CrossEntropyLoss()
	
	def __call__(self, outputs, targets):
		if isinstance(outputs, dict): outputs = outputs[P.KEY_CLASS_SCORES]
		if isinstance(targets, dict): targets = targets[P.KEY_LABEL_TARGETS]
		return self.crossent_loss(outputs, targets)
		

# Criterion manager for cross entropy loss
class CrossEntMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return CrossEntMetric()
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "cross-entropy"