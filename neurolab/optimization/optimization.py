# Optimization manager base class
class OptimManager:
	def __init__(self, config):
		self.config = config
	
	def get_optimizer(self, net):
		raise NotImplementedError

# Scheduling manager base class
class SchedManager:
	def __init__(self, config):
		self.config = config
	
	def get_scheduler(self, optimizer):
		raise NotImplementedError

# Criterion manager base class
class MetricManager:
	def __init__(self, config):
		self.config = config
	
	# Returns a callable implementing the logic for evaluating the desired objective
	def get_metric(self):
		raise NotImplementedError
	
	# Returns True if the objective is of type HB and False if LB
	def higher_is_better(self):
		raise NotImplementedError
	
	# Returns the name of this objective as string
	def get_name(self):
		return "result"