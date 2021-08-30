from torchvision.datasets import STL10
from torch.utils.data import Subset
import random

from .data import DataManager
from .. import params as P


# Data manager class for STL10
class STL10DataManager(DataManager):
	def __init__(self, config):
		self.STL10_USE_LBL_SPLIT = config.CONFIG_OPTIONS.get(P.KEY_DS_USE_LBL_SPLIT, True)
		self.STL10_NUM_LABELED = 5000
		self.STL10_NUM_UNLABELED = 100000
		self.idx_labeled = list(range(self.STL10_NUM_LABELED))
		self.idx_unlabeled = list(range(self.STL10_NUM_UNLABELED))
		
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'stl10',
			P.KEY_DS_TRN_SET_SIZE: self.STL10_NUM_LABELED if self.STL10_USE_LBL_SPLIT else self.STL10_NUM_UNLABELED + self.STL10_NUM_LABELED,
			P.KEY_DS_VAL_SET_SIZE: 8000,
			P.KEY_DS_TST_SET_SIZE: 8000,
			P.KEY_DS_INPUT_SHAPE: (3, 96, 96),
			P.KEY_DS_NUM_CLASSES: 10
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.idx_labeled)
		random.shuffle(self.idx_unlabeled)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None:
			indices = (list(range(self.STL10_NUM_UNLABELED, self.STL10_NUM_UNLABELED + self.NUM_TRN_SAMPLES)) + ([] if self.TOT_TRN_SAMPLES <= self.NUM_TRN_SAMPLES else list(range(self.TOT_TRN_SAMPLES - self.NUM_TRN_SAMPLES)))) if self.STL10_USE_LBL_SPLIT \
				else (range(self.STL10_NUM_UNLABELED, self.STL10_NUM_UNLABELED + self.TOT_TRN_SAMPLES) if self.TOT_TRN_SAMPLES <= self.STL10_NUM_LABELED
				      else (list(range(self.STL10_NUM_UNLABELED, self.STL10_NUM_UNLABELED + self.STL10_NUM_LABELED)) + list(range(self.TOT_TRN_SAMPLES - self.STL10_NUM_LABELED))))
			split = 'train' if (self.TOT_TRN_SAMPLES <= self.NUM_TRN_SAMPLES and self.STL10_USE_LBL_SPLIT) or (self.TOT_TRN_SAMPLES <= self.STL10_NUM_LABELED and not self.STL10_USE_LBL_SPLIT) else 'train+unlabeled'
			shuffled_indices = [self.idx_labeled[i] for i in indices] if split=='train' \
				else [self.idx_unlabeled[i] if i < self.STL10_NUM_UNLABELED else self.STL10_NUM_UNLABELED + self.idx_labeled[i-self.STL10_NUM_UNLABELED] for i in indices]
			return Subset(STL10(root=self.DATASET_FOLDER, split=split, download=True, transform=transform), shuffled_indices)
		indices = range(num_samples) if self.STL10_USE_LBL_SPLIT \
			else (range(self.STL10_NUM_UNLABELED, num_samples) if num_samples <= self.STL10_NUM_LABELED
			      else (list(range(self.STL10_NUM_UNLABELED, self.STL10_NUM_UNLABELED + self.STL10_NUM_LABELED)) + list(range(num_samples - self.STL10_NUM_LABELED))))
		split = 'train' if self.STL10_USE_LBL_SPLIT or (num_samples <= self.STL10_NUM_LABELED and not self.STL10_USE_LBL_SPLIT)  else 'train+unlabeled'
		shuffled_indices = [self.idx_labeled[i] for i in indices] if split=='train' \
			else [self.idx_unlabeled[i] if i < self.STL10_NUM_UNLABELED else self.STL10_NUM_UNLABELED + self.idx_labeled[i-self.STL10_NUM_UNLABELED] for i in indices]
		return Subset(STL10(root=self.DATASET_FOLDER, split=split, download=True, transform=transform), shuffled_indices)
		
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		validate_on_test = self.TRN_SET_SIZE - (self.NUM_TRN_SAMPLES if self.STL10_USE_LBL_SPLIT else self.TOT_TRN_SAMPLES) >= self.NUM_VAL_SAMPLES
		if validate_on_test:
			# If there are enough samples left in the training set, use them for validation
			indices = range(self.STL10_NUM_LABELED - num_samples, self.STL10_NUM_LABELED) if self.STL10_USE_LBL_SPLIT \
				else (range(self.STL10_NUM_UNLABELED - num_samples, self.STL10_NUM_UNLABELED) if num_samples <= self.STL10_NUM_UNLABELED
				      else (list(range(self.STL10_NUM_UNLABELED + self.STL10_NUM_LABELED - (num_samples - self.STL10_NUM_UNLABELED), self.STL10_NUM_UNLABELED + self.STL10_NUM_LABELED)) + list(range(self.STL10_NUM_UNLABELED))))
			split = 'train' if self.STL10_USE_LBL_SPLIT else 'train+unlabeled'
			shuffled_indices = [self.idx_labeled[i] for i in indices] if split=='train' \
				else [self.idx_unlabeled[i] if i < self.STL10_NUM_UNLABELED else self.STL10_NUM_UNLABELED + self.idx_labeled[i-self.STL10_NUM_UNLABELED] for i in indices]
			return Subset(STL10(root=self.DATASET_FOLDER, split=split, download=True, transform=transform), shuffled_indices)
		# Validate directly on test set otherwise
		return Subset(STL10(root=self.DATASET_FOLDER, split='test', download=True, transform=transform), range(num_samples))
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(STL10(root=self.DATASET_FOLDER, split='test', download=True, transform=transform), range(num_samples))

