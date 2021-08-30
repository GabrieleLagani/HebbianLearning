from torchvision.datasets.utils import extract_archive
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, WeightedRandomSampler
import random
import os

from ..utils import download_large_file_from_drive
from .data import DataManager
from .. import params as P



# Caltech101 Dataset
class Caltech101(ImageFolder):
	def __init__(self, root, download=False, **kwargs):
		self.root = root
		self.download = download
		self.CALTECH101_DRIVE_ID = '137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp'
		self.CALTECH101_ZIP_FILE = '101_ObjectCategories.tar.gz'
		self.CALTECH101_ZIP_PATH = os.path.join(self.root, self.CALTECH101_ZIP_FILE)
		self.CALTECH101_FOLDER = os.path.join(self.root, '101_ObjectCategories')
		
		if not os.path.exists(self.CALTECH101_FOLDER):
			# Extract Caltech101 zip file
			if not os.path.exists(self.CALTECH101_ZIP_PATH):
				if not self.download: raise(FileNotFoundError("Dataset files not found"))
				print("Downloading {} file...".format(self.CALTECH101_ZIP_FILE))
				download_large_file_from_drive(self.CALTECH101_DRIVE_ID, self.CALTECH101_ZIP_PATH)
				print("Done!")
			print("Extracting {} file...".format(self.CALTECH101_ZIP_FILE))
			extract_archive(self.CALTECH101_ZIP_PATH, self.root)
			print("Done!")
		
		super(Caltech101, self).__init__(self.CALTECH101_FOLDER, **kwargs)
		
		self.samples_per_class = {k: self.targets.count(k) for k in self.class_to_idx.values()}

# Data manager class for Caltech101
class Caltech101DataManager(DataManager):
	def __init__(self, config):
		self.CALTECH101_SIZE = 9144
		self.indices = list(range(self.CALTECH101_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'caltech101',
			P.KEY_DS_TRN_SET_SIZE: 8144,
			P.KEY_DS_VAL_SET_SIZE: 1000,
			P.KEY_DS_TST_SET_SIZE: 1000,
			P.KEY_DS_INPUT_SHAPE: (3, 224, 224),
			P.KEY_DS_NUM_CLASSES: 102
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.indices)
	
	def get_sampler(self, split):
		# split is a Subset of Caltech101. split.dataset is Caltech101. From this we can access the number of samples
		# per class, to be used for weighted random sampling, together with the length of the split, to determine how
		# many samples to fetch.
		return WeightedRandomSampler([w for w in split.dataset.samples_per_class.values()], len(split))
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(Caltech101(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		if self.TRN_SET_SIZE - self.TOT_TRN_SAMPLES >= num_samples:
			# If there are enough samples left in the training set, use them for validation
			return Subset(Caltech101(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE - num_samples:self.TRN_SET_SIZE])
		# Validate directly on test set otherwise
		return Subset(Caltech101(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE:self.TRN_SET_SIZE + num_samples])
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(Caltech101(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE:self.TRN_SET_SIZE + num_samples])

