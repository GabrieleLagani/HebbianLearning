from ..optimization import MetricManager
from neurolab import params as P


# Evaluate outputs on a retrieval experiment basis: outputs must contain percentage of retrieved elements for each class.
# Return percentage of retrieved elements of the correct class (average precision). The experiment class is going to
# average this result over a batch of data, thus finally obtaining a Mean Average Precision (MAP) score.
class PrecMetric:
	def __call__(self, outputs, targets):
		if isinstance(outputs, dict): outputs = outputs[P.KEY_CLASS_SCORES]
		if isinstance(targets, dict): targets = targets[P.KEY_LABEL_TARGETS]
		return outputs.gather(dim=1, index=targets.view(-1, 1)).mean().item()

# Criterion manager for MAP
class PrecMetricManager(MetricManager):
	def __init__(self, config):
		super().__init__(config)
	
	def get_metric(self):
		return PrecMetric()

	def higher_is_better(self):
		return True
	
	def get_name(self):
		return "precision"