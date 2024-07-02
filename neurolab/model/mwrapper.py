import torch
import torch.nn as nn

from .. import params as P
from .model import Model


# Standard wrapper for neurolab models to provide the a standard signature as a LayerSet, which is the one expected in Experiment classes.
class ModuleWrapper(Model):
	def __init__(self, config, input_shape=None):
		super(ModuleWrapper, self).__init__(config, input_shape)
	
	# Here we define the flow of information through the network
	def wrapped_forward(self, x):
		raise NotImplementedError
	
	# Here we define the interface between forward signature expected by neurolab, and the forward method of the wrapped module
	def forward(self, x, queries=None, regroup=True):
		x = self.wrapped_forward(x)
		
		# Prepare keys for output dictionary
		single_query = not isinstance(queries, (list, tuple, set))
		defaults = [None]
		allkeys = list(x.keys()) if isinstance(x, dict) else defaults
		queries = {q: self.specify_key(q, allkeys, defaults) for q in ({queries} if single_query else set(queries))}
		spec_queries = {k for v in queries.values() for k in v}
		out = {}
		
		# Collect outputs in a dictionaries
		if not isinstance(x, dict):
			unmatched_keys = {k for k in spec_queries if k is not None and k != ''}
			if len(unmatched_keys) > 0:
				ke = KeyError(str(queries))
				ke.unmatched_keys = unmatched_keys
				raise ke
			out = {n: {k: x for k in queries[n]} for n in queries}
		else:
			unmatched_keys = spec_queries.difference(x.keys())
			if len(unmatched_keys) > 0:
				ke = KeyError(str(unmatched_keys))
				ke.unmatched_keys = unmatched_keys
				raise ke
			out = {n: {k: x[k] for k in queries[n]} for n in queries}
		
		# Group keys by query in output dictionary if required
		if regroup: out = {k: out[n][k] for n in out for k in out[n]}
		
		# Unwrap output from dictionary if single output is required
		if single_query: return out if len(out) > 1 else (out[list(out.keys())[0]] if len(out) == 1 else None)
		
		return out


# Standard simplified wrapper for neurolab models to provide the a standard signature as a LayerSet, which is the one expected in Experiment classes.
class SimpleWrapper(ModuleWrapper):
	def __init__(self, config, input_shape=None):
		super().__init__(config, input_shape)
		self.module = self.wrapped_init(config, input_shape)
	
	# Execute initialization logic for wrapper module and return initialized module
	def wrapped_init(self, config, input_shape=None):
		raise NotImplementedError
	
	def state_dict(self):
		return self.module.state_dict()
	
	def load_state_dict(self, state_dict, strict=True):
		return self.module.load_state_dict(state_dict, strict=strict)
	
	def wrapped_forward(self, x):
		return self.module(x)
	
	# Function for setting teacher signal for supervised local update rules
	def set_teacher_signal(self, y):
		for m in self.module.children():
			if hasattr(m, 'set_teacher_signal'): m.set_teacher_signal(y)
	
	# Function for applying local update steps
	def local_update(self):
		for m in self.module.children():
			if hasattr(m, 'local_update'): m.local_update()
	
	# Function to reset state of internal scheduling policies inside the model
	def reset_internal_sched_state(self):
		for m in self.module.children():
			if hasattr(m, 'reset_internal_sched_state'): m.reset_internal_sched_state()
	
	