from torchvision.datasets import MNIST
from torch.utils.data import Subset
import random

from .data import DataManager
from .. import params as P


# Data manager class for MNIST
class MNISTDataManager(DataManager):
	def __init__(self, config):
		self.MNIST_TRN_SET_SIZE = 60000
		self.indices = list(range(self.MNIST_TRN_SET_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'mnist',
			P.KEY_DS_TRN_SET_SIZE: self.MNIST_TRN_SET_SIZE,
			P.KEY_DS_VAL_SET_SIZE: 10000,
			P.KEY_DS_TST_SET_SIZE: 10000,
			P.KEY_DS_INPUT_SHAPE: (1, 28, 28),
			P.KEY_DS_NUM_CLASSES: 10
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.indices)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(MNIST(root=self.DATASET_FOLDER, train=True, download=True, transform=transform), self.indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		if self.TRN_SET_SIZE - self.TOT_TRN_SAMPLES >= num_samples:
			# If there are enough samples left in the training set, use them for validation
			return Subset(MNIST(root=self.DATASET_FOLDER, train=True, download=True, transform=transform), self.indices[self.TRN_SET_SIZE - num_samples:self.TRN_SET_SIZE])
		# Validate directly on test set otherwise
		return Subset(MNIST(root=self.DATASET_FOLDER, train=False, download=True, transform=transform), range(num_samples))
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(MNIST(root=self.DATASET_FOLDER, train=False, download=True, transform=transform), range(num_samples))

