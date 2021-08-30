from torchvision.datasets.utils import extract_archive, download_url
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset
import random
import os
import struct
import numpy as np
from PIL import Image

from .data import DataManager
from .. import params as P



# SmallNORB Dataset
class SmallNORB(VisionDataset):
	def __init__(self, root, split='train', download=False, transform=None, target_transform=None):
		self.root = root
		self.download = download
		self.split = split
		self.transform = transform
		self.target_transform = target_transform
		self.SMALLNORB_URLS = {
			'train-dat' : 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz',
			'train-cat' : 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz',
			'train-info': 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz',
			'test-dat'  : 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz',
			'test-cat'  : 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz',
			'test-info' : 'https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz',
		}
		self.SMALLNORB_ZIP_FILES =  {
			'train-dat' : 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz',
			'train-cat' : 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz',
			'train-info': 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz',
			'test-dat'  : 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz',
			'test-cat'  : 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz',
			'test-info' : 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz',
		}
		self.SMALLNORB_ZIP_PATHS = {id: os.path.join(self.root, self.SMALLNORB_ZIP_FILES[id]) for id in self.SMALLNORB_ZIP_FILES}
		self.SMALLNORB_MAT_FILES =  {
			'train-dat' : 'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat',
			'train-cat' : 'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat',
			'train-info': 'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat',
			'test-dat'  : 'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat',
			'test-cat'  : 'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat',
			'test-info' : 'smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat',
		}
		self.SMALLNORB_MAT_PATHS = {id: os.path.join(self.root, self.SMALLNORB_MAT_FILES[id]) for id in self.SMALLNORB_MAT_FILES}
		
		self.SMALLNORB_DAT_HEADER_SIZE = 24
		self.SMALLNORB_CAT_HEADER_SIZE = 20
		self.SMALLNORB_IMG_HEIGHT = 96
		self.SMALLNORB_IMG_WIDTH = 96
		self.SMALLNORB_IMG_SIZE = self.SMALLNORB_IMG_HEIGHT * self.SMALLNORB_IMG_WIDTH
		self.SMALLNORB_IMG_MULTIPLICITY = 2
		self.SMALLNORB_CATEGORY_SIZE = 4
		
		for id in self.SMALLNORB_MAT_PATHS:
			# Extract SmallNORB zip file
			if not os.path.exists(self.SMALLNORB_MAT_PATHS[id]):
				if not os.path.exists(self.SMALLNORB_ZIP_PATHS[id]):
					if not self.download: raise(FileNotFoundError("Dataset files not found"))
					download_url(self.SMALLNORB_URLS[id], self.root, self.SMALLNORB_ZIP_FILES[id])
				print("Extracting {} file...".format(self.SMALLNORB_ZIP_FILES[id]))
				extract_archive(self.SMALLNORB_ZIP_PATHS[id], self.root)
				print("Done!")
		
		super(SmallNORB, self).__init__(self.root, transform=self.transform, target_transform=self.target_transform)
	
	def __getitem__(self, item):
		if item >= len(self): raise IndexError("Index {} out of bound for dataset of length {}".format(item, len(self)))
		dat_file_path = self.SMALLNORB_MAT_PATHS['train-dat' if self.split == 'train' else 'test-dat']
		cat_file_path = self.SMALLNORB_MAT_PATHS['train-cat' if self.split == 'train' else 'test-cat']
		image = None
		category = None
		with open(dat_file_path, mode='rb') as f:
			f.seek(self.SMALLNORB_DAT_HEADER_SIZE + item * self.SMALLNORB_IMG_SIZE * self.SMALLNORB_IMG_MULTIPLICITY)
			image = struct.unpack('<' + self.SMALLNORB_IMG_SIZE * 'B', f.read(self.SMALLNORB_IMG_SIZE))
			image = np.reshape(image, newshape=(self.SMALLNORB_IMG_HEIGHT, self.SMALLNORB_IMG_WIDTH))
			image = np.uint8(image)
			image = Image.fromarray(image)
		with open(cat_file_path, mode='rb') as f:
			f.seek(self.SMALLNORB_CAT_HEADER_SIZE + item * self.SMALLNORB_CATEGORY_SIZE)
			category = struct.unpack('<i', f.read(4))[0]
		if self.transform is not None:
			image = self.transform(image)
		if self.target_transform is not None:
			category = self.target_transform(category)
		return image, category
	
	def __len__(self):
		return 24300
	
# Data manager class for SmallNORB
class SmallNORBDataManager(DataManager):
	def __init__(self, config):
		self.SMALLNORB_SIZE = 24300
		self.trn_indices = list(range(self.SMALLNORB_SIZE))
		self.tst_indices = list(range(self.SMALLNORB_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'smallnorb',
			P.KEY_DS_TRN_SET_SIZE: self.SMALLNORB_SIZE,
			P.KEY_DS_VAL_SET_SIZE: self.SMALLNORB_SIZE,
			P.KEY_DS_TST_SET_SIZE: self.SMALLNORB_SIZE,
			P.KEY_DS_INPUT_SHAPE: (1, 96, 96),
			P.KEY_DS_NUM_CLASSES: 5
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.trn_indices)
		random.shuffle(self.tst_indices)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(SmallNORB(root=self.DATASET_FOLDER, download=True, split='train', transform=transform), self.trn_indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		# Validate on a random subset of test split
		return Subset(SmallNORB(root=self.DATASET_FOLDER, download=True, split='test', transform=transform), self.tst_indices[self.SMALLNORB_SIZE - num_samples: self.SMALLNORB_SIZE])
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		# Test on a random subset of test split
		return Subset(SmallNORB(root=self.DATASET_FOLDER, download=True, split='test', transform=transform), self.tst_indices[0:num_samples])

