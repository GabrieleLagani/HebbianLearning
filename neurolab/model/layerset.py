import torch
import torch.nn as nn

from .. import params as P
from .model import Model
from .mwrapper import ModuleWrapper


# Custom Sequential module class for naming layers, even assigning synonyms to the same layer, and extracting outputs from specific layers.
class LayerSet(Model):
	def __init__(self, config, input_shape=None):
		super(LayerSet, self).__init__(config, input_shape)
		
		# This setting can be adjusted in subclasses. If True it stops processing early as soon as all required outputs
		# have been computed. Otherwise it will continue until the end.
		self.earlybreak = True
		
		# Layer sequence with corresponding names
		self.layers = []
		self.layer_names = []
		self.layer_labels = []
		self.layer_queries = []
		self.layer_remaps = []
		self.layer_lookup = {}
	
	# Add a named layer to the layer sequence
	def add_layer(self, layer, names, labels=None, queries=None, remaps=None):
		names = set(names) if isinstance(names, (list, tuple, set)) else {names}
		if labels is None: labels = set()
		labels = set(labels) if isinstance(labels, (list, tuple, set)) else {labels}
		labels = labels.union(names)
		
		dups = set()
		for n in self.layer_names: dups = dups.union(labels.intersection(n))
		if len(dups) > 0: raise KeyError("Keys " + str(dups) + " already in use.")
		
		if any(l is None or '*' in l or '.' in l for l in labels): raise KeyError("Invalid names for layer: " + str(labels))
		
		if not isinstance(layer, Model): layer = ModuleWrapper(layer, config=self.config, input_shape=None)
		
		layer_idx = len(self.layers)
		for l in labels:
			if l not in self.layer_lookup: self.layer_lookup[l] = set()
			self.layer_lookup[l].add(layer_idx)
		self.layers.append(layer)
		self.layer_names.append(names)
		self.layer_labels.append(labels)
		self.layer_queries.append(queries)
		self.layer_remaps.append(remaps)
		for n in names: self.add_module(n, layer) # Register module within Pytorch
	
	# Checks if some layer with a given label exists
	def layer_exists(self, label):
		return label in self.layer_lookup
	
	# Checks if a given label is also a name of some layer
	def is_name(self, label):
		if label not in self.layer_lookup: return False
		if len(self.layer_lookup[label]) != 1: return False
		return label in self.layer_names[list(self.layer_lookup[label])[0]]
	
	# Returns all layers from a given label as a dictionary containing layer name as key and corresponding layer as value.
	def get_layers(self, label):
		if label not in self.layer_lookup: return {}
		return {list(self.layer_names[l])[0]: self.layers[l] for l in self.layer_lookup[label]}
	
	# Looks up all the labels associated to all layers matching a given label. Returns a dictionary with one entry for each layer,
	# where the key is the layer name and the value is the corresponding set of labels.
	def lookup_labels(self, label):
		if label not in self.layer_lookup: return {}
		return {list(self.layer_names[l])[0]: self.layer_labels[l] for l in self.layer_lookup[label]}
	
	# Looks up all the names associated to all layers matching a given label. Returns a dictionary with one entry for each layer,
	# where the key is the layer primary name and the value is the corresponding set of names.
	def lookup_names(self, label):
		if label not in self.layer_lookup: return {}
		return {list(self.layer_names[l])[0]: self.layer_names[l] for l in self.layer_lookup[label]}
	
	# Returns default input keys when no input key is provided. This corresponds to the first layer, but this behavior can be overridden.
	def get_default_input_keys(self):
		return [list(self.layer_names[0])[0]]
	
	# Returns default output keys when no output key is provided. This corresponds to the last layer, but this behavior can be overridden.
	def get_default_output_keys(self):
		return [list(self.layer_names[-1])[0]]
		
	# Return default remapping of keys from a given layer for routing outputs to subsequent layers. By default this gives
	# a feedforward processing style, but this behavior can be overridden.
	def get_default_layer_remap(self, i):
		return list(self.layer_names[i+1])[0] if i+1 < len(self.layer_names) else None
	
	# Here we define the flow of information through the network
	def forward(self, x, queries=None, regroup=True):
		allkeys = list(self.layer_lookup.keys()) # This contains all the layer labels
		defaults = self.get_default_output_keys() # This contains default output keys
		defaults = list(defaults) if isinstance(defaults, (list, tuple, set)) else [defaults]
		inp_defaults = self.get_default_input_keys() # This contains the default input keys
		inp_defaults = list(inp_defaults) if isinstance(inp_defaults, (list, tuple, set)) else [inp_defaults]
		
		# Prepare input as a dictionary with keys referring to target layers
		if not isinstance(x, dict): x = {None: x}
		x = {n: x[k] for k in x for h in self.specify_key(k, allkeys, inp_defaults) for n in (self.lookup_names(h) if self.layer_exists(h) else {h})}
		allkeys += list(x.keys())
		
		# Argument queries is transformed into a dictionary mapping each query to a set of keys that must be returned for that query.
		single_query = not isinstance(queries, (list, tuple, set)) # Denotes whether a single query is provided rather than a dictionary of queries
		queries = {q: self.specify_key(q, allkeys, defaults) for q in ({queries} if single_query else set(queries))}
		spec_queries = {k for v in queries.values() for k in v}
		
		# Extract actual keys from keys argument
		split_queries = {}
		for k in spec_queries:
			prefix, suffix = k, None
			if '.' in k: prefix, suffix = k.split('.', 1)
			if prefix not in split_queries: split_queries[prefix] = set()
			split_queries[prefix].add(suffix)
		remaining_keys = set(split_queries.keys()).intersection(allkeys)
		unmatched_keys = remaining_keys.difference(allkeys)
		out = {k: x[k] for k in unmatched_keys if k in x and all(h is None or h == '' for h in split_queries[k])}
		unmatched_keys = unmatched_keys.difference(out.keys())
		if len(unmatched_keys) > 0:
			ke = KeyError(str(unmatched_keys))
			ke.unmatched_keys = unmatched_keys
			raise ke
		remaining_keys = {h for k in remaining_keys for h in self.lookup_names(k)}
		
		# Process data through layers, and add results to the output dictionary. Terminate when outputs have been collected for all keys.
		for i in range(len(self.layers)):
			if len(remaining_keys) == 0 and self.earlybreak: break
			
			# Prepare keys for nested output collection
			nested_queries = {k for l in self.layer_labels[i] if l in split_queries for k in split_queries[l]}.union(
				self.layer_queries[i] if isinstance(self.layer_queries[i], (list, tuple, set)) else {self.layer_queries[i]})
			
			# Prepare inputs
			x1 = {}
			for k in x:
				prefix, suffix = k, None
				if '.' in k: prefix, suffix = k.split('.', 1)
				if prefix in self.layer_names[i]: x1[suffix] = x.pop(k)
			if len(x1) == 1 and None in x1: x1 = x1[None]
			
			# Compute layer outputs
			try:
				x2 = self.layers[i](x1, queries=nested_queries, regroup=False)
			except KeyError as ke:
				if hasattr(ke, 'unmatched_keys'):
					unmatched_keys = {l + '.' + k for l in self.layer_labels[i] if l in split_queries for k in ke.unmatched_keys if k in split_queries[l]}
					ke = KeyError(str(unmatched_keys))
					ke.unmatched_keys = unmatched_keys
				raise ke
			
			# Collect outputs in dictionaries
			for l in self.layer_labels[i]:
				if l in split_queries:
					for k in split_queries[l]:
						x2_k = x2.get(k, {}) if isinstance(x2, dict) else x2
						out_key = l if k is None else (l + '.' + k)
						if out_key not in out: out[out_key] = {}
						if not isinstance(x2_k, dict): x2_k = {None: x2_k}
						for h in x2_k:
							comp_key = l if h is None else (l + '.' + h)
							if self.is_name(l):
								out[out_key][comp_key] = x2_k[h]
							else:
								if comp_key not in out[out_key]: out[out_key][comp_key] = {}
								for n in self.layer_names[i]: out[out_key][comp_key][n] = x2_k[h]
				if l in remaining_keys: remaining_keys.remove(l)
			layer_remap = self.layer_remaps[i] if self.layer_remaps[i] is not None else self.get_default_layer_remap(i)
			if not isinstance(layer_remap, (list, tuple, set, dict)): layer_remap = {layer_remap}
			for n in layer_remap:
				if n is not None and n != '':
					x_n = (x2.get(layer_remap[n], None) if isinstance(x2, dict) else None) if isinstance(layer_remap, dict) else x2
					for m in [h for k in self.specify_key(n, allkeys, defaults) for h in self.lookup_names(k)]:
						if isinstance(x_n, dict) and len(x_n) > 0:
							for k in x_n: x[m + '.' + k] = x_n[k]
						else: x[m] = x_n
		
		# Group keys by query in output dictionary if required
		out = {n: {h: out[k][h] for k in queries[n] for h in out[k]} for n in queries}
		if regroup: out = {k: out[n][k] for n in out for k in out[n]}
		
		# Unwrap output from dictionary if single output is required
		if single_query: out = out if len(out) > 1 else (out[list(out.keys())[0]] if len(out) == 1 else None)
		
		return out
	
	# Function for setting teacher signal for supervised local update rules
	def set_teacher_signal(self, y):
		if self.DEEP_TEACHER_SIGNAL:
			for l in self.layers[:-1]:
				if hasattr(l, 'set_teacher_signal'): l.set_teacher_signal(y)
		if hasattr(self.layers[-1], 'set_teacher_signal'): self.layers[-1].set_teacher_signal(y)
	
	