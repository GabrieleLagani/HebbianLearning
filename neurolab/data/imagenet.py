from torchvision.datasets import ImageNet
from torch.utils.data import Subset
import random

from .data import DataManager
from .. import params as P


# Data manager class for ImageNet
class ImageNetDataManager(DataManager):
	def __init__(self, config):
		self.IMAGENET_TRN_SET_SIZE = 1281167
		self.indices = list(range(self.IMAGENET_TRN_SET_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'imagenet',
			P.KEY_DS_TRN_SET_SIZE: self.IMAGENET_TRN_SET_SIZE,
			P.KEY_DS_VAL_SET_SIZE: 50000,
			P.KEY_DS_TST_SET_SIZE: 50000,
			P.KEY_DS_INPUT_SHAPE: (3, 224, 224),
			P.KEY_DS_NUM_CLASSES: 1000
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.indices)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(ImageNet(root=self.DATASET_FOLDER, split='train', transform=transform), self.indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		if self.TRN_SET_SIZE - self.TOT_TRN_SAMPLES >= num_samples:
			# If there are enough samples left in the training set, use them for validation
			return Subset(ImageNet(root=self.DATASET_FOLDER, split='train', transform=transform), self.indices[self.TRN_SET_SIZE - num_samples:self.TRN_SET_SIZE])
		# Validate directly on test set otherwise
		return Subset(ImageNet(root=self.DATASET_FOLDER, split='val', transform=transform), range(num_samples))
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(ImageNet(root=self.DATASET_FOLDER, split='val', transform=transform), range(num_samples))

