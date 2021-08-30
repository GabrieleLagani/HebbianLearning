import torch.nn as nn

from ..optimization import MetricManager
from neurolab import params as P
from neurolab import utils


# Wrapper around Pytorch MSELoss criterion
class MSEMetric:
	def __init__(self):
		self.mse_loss = nn.MSELoss()
	
	def __call__(self, outputs, targets):
		if isinstance(outputs, dict): outputs = outputs[P.KEY_CLASS_SCORES]
		if isinstance(targets, dict): targets = targets[P.KEY_LABEL_TARGETS]
		return self.mse_loss(outputs, utils.dense2onehot(targets, outputs.size(1)))
		

# Criterion manager for MSE loss
class MSEMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return MSEMetric()
	
	def higher_is_better(self):
		return False
	
	def get_name(self):
		return "mse"