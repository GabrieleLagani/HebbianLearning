import torch

from ..optimization import MetricManager
from neurolab import params as P


# Return number of correctly classified samples given network outputs and target labels. The experiment class is then
# going to average over a batch of data, thus finally obtaining an accuracy score.
class AccMetric:
	def __call__(self, outputs, targets):
		if isinstance(outputs, dict): outputs = outputs[P.KEY_CLASS_SCORES]
		if isinstance(targets, dict): targets = targets[P.KEY_LABEL_TARGETS]
		# The predicted class is the argmax of the output tensor along dimension 1 (dimension 0 is the batch dimension)
		_, pred = torch.max(outputs, 1)
		return (pred == targets).float().mean().item()

# Criterion manager for accuracy
class AccMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return AccMetric()
	
	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "accuracy"