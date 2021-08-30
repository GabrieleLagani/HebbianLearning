from torchvision.datasets.utils import extract_archive, download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import random
import os
import shutil

from .data import DataManager
from .. import params as P



# ETH80 Dataset
class ETH80(ImageFolder):
	def __init__(self, root, download=False, **kwargs):
		self.root = root
		self.download = download
		self.ETH80_URL = 'https://github.com/Kai-Xuan/ETH-80/archive/master.zip'
		self.ETH80_ZIP_FILE = 'ETH-80-master.zip'
		self.ETH80_ZIP_PATH = os.path.join(self.root, self.ETH80_ZIP_FILE)
		self.ETH80_FOLDER = os.path.join(self.root, 'ETH-80-master')
		self.ETH80_FOLDER_ORGANIZED = os.path.join(self.root, 'ETH-80-organized')
		
		if not os.path.exists(self.ETH80_FOLDER):
			# Extract ETH80 zip file
			if not os.path.exists(self.ETH80_ZIP_PATH):
				if not self.download: raise(FileNotFoundError("Dataset files not found"))
				download_url(self.ETH80_URL, self.root, self.ETH80_ZIP_FILE)
			print("Extracting {} file...".format(self.ETH80_ZIP_FILE))
			extract_archive(self.ETH80_ZIP_PATH, self.root)
			print("Done!")
		 
		# Organize images in a directory structure consistent with that required by ImageFolder. Also, ignore "map" images.
		if not os.path.exists(self.ETH80_FOLDER_ORGANIZED):
			print("Organizing dataset images...")
			os.makedirs(self.ETH80_FOLDER_ORGANIZED)
			for i in range(1, 9):
				dest = os.path.join(self.ETH80_FOLDER_ORGANIZED, str(i))
				os.makedirs(dest)
				for j in range(1, 11):
					src = os.path.join(self.ETH80_FOLDER, str(i), str(j))
					for file in os.listdir(src):
						if file != 'maps': shutil.copy(os.path.join(src, file), os.path.join(dest, file))
			print("Done!")
			
		super(ETH80, self).__init__(self.ETH80_FOLDER_ORGANIZED, **kwargs)

# Data manager class for ETH80
class ETH80DataManager(DataManager):
	def __init__(self, config):
		self.ETH80_SIZE = 3280
		self.indices = list(range(self.ETH80_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'eth80',
			P.KEY_DS_TRN_SET_SIZE: 2880,
			P.KEY_DS_VAL_SET_SIZE: 400,
			P.KEY_DS_TST_SET_SIZE: 400,
			P.KEY_DS_INPUT_SHAPE: (3, 96, 96),
			P.KEY_DS_NUM_CLASSES: 8
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.indices)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(ETH80(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		if self.TRN_SET_SIZE - self.TOT_TRN_SAMPLES >= num_samples:
			# If there are enough samples left in the training set, use them for validation
			return Subset(ETH80(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE - num_samples:self.TRN_SET_SIZE])
		# Validate directly on test set otherwise
		return Subset(ETH80(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE:self.TRN_SET_SIZE + num_samples])
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(ETH80(root=self.DATASET_FOLDER, download=True, transform=transform), self.indices[self.TRN_SET_SIZE:self.TRN_SET_SIZE + num_samples])

