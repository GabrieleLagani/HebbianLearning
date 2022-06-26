from time import time

import torch


class Timer:
	def __init__(self):
		self.t0 = None
	
	def synch(self):
		for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)
		
	def tick(self):
		self.synch()
		return time()
	
	def start(self):
		self.t0 = self.tick()
		return self
	
	def record(self, label=''):
		t = self.tick()
		print(label + ': ' + str(t - self.t0))
		self.t0 = t

