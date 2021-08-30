from torchvision.datasets.utils import extract_archive
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
import random
import os
import shutil

from .data import DataManager
from .. import params as P



# Tiny ImageNet Dataset
class TinyImageNet(ImageFolder):
	def __init__(self, root, split='train', **kwargs):
		self.root = root
		self.split = split
		self.TINYIMAGENET_ZIP_FILE = 'tiny-imagenet-200.zip'
		self.TINYIMAGENET_ZIP_PATH = os.path.join(self.root, self.TINYIMAGENET_ZIP_FILE)
		self.TINYIMAGENET_FOLDER = os.path.join(self.root, 'tiny-imagenet-200')
		self.TINYIMAGENET_TRAIN_FOLDER = os.path.join(self.TINYIMAGENET_FOLDER, 'train')
		self.TINYIMAGENET_VAL_FOLDER = os.path.join(self.TINYIMAGENET_FOLDER, 'val')
		self.TINYIMAGENET_VAL_TARGET_FILE = os.path.join(self.TINYIMAGENET_VAL_FOLDER, 'val_annotations.txt')
		self.TINYIMAGENET_VAL_IMAGES_FOLDER = os.path.join(self.TINYIMAGENET_VAL_FOLDER, 'images')
		self.TINYIMAGENET_VAL_ORGANIZED_IMAGES_FOLDER = os.path.join(self.TINYIMAGENET_VAL_FOLDER, 'organized_images')
		
		if not os.path.exists(self.TINYIMAGENET_FOLDER):
			# Extract tinyimagenet zip file
			if not os.path.exists(self.TINYIMAGENET_ZIP_PATH):
				raise FileNotFoundError("Please download the {} file and place it in the {} folder".format(self.TINYIMAGENET_ZIP_FILE, self.root))
			print("Extracting {} file...".format(self.TINYIMAGENET_ZIP_FILE))
			extract_archive(self.TINYIMAGENET_ZIP_PATH, self.root)
			print("Done!")
		
		# Organize validation set as required by image folder, if necessary.
		if (self.split != 'train') and not os.path.exists(self.TINYIMAGENET_VAL_ORGANIZED_IMAGES_FOLDER):
			print("Organizing dataset images...")
			with open(self.TINYIMAGENET_VAL_TARGET_FILE) as target_file: lines = target_file.readlines()
			images = [os.path.join(self.TINYIMAGENET_VAL_IMAGES_FOLDER, l.split("\t")[0]) for l in lines]
			wnids = [l.split("\t")[1] for l in lines]
			for wnid in set(wnids): os.makedirs(os.path.join(self.TINYIMAGENET_VAL_ORGANIZED_IMAGES_FOLDER, wnid), exist_ok=True)
			for wnid, img_file in zip(wnids, images): shutil.copy(img_file, os.path.join(self.TINYIMAGENET_VAL_ORGANIZED_IMAGES_FOLDER, wnid, os.path.basename(img_file)))
			print("Done!")
		
		super(TinyImageNet, self).__init__(self.TINYIMAGENET_TRAIN_FOLDER if self.split == 'train' else self.TINYIMAGENET_VAL_ORGANIZED_IMAGES_FOLDER, **kwargs)
		

# Data manager class for TinyImageNet
class TinyImageNetDataManager(DataManager):
	def __init__(self, config):
		self.TINYIMAGENET_TRN_SET_SIZE = 100000
		self.indices = list(range(self.TINYIMAGENET_TRN_SET_SIZE))
		super().__init__(config)
	
	def get_dataset_metadata(self):
		return {
			P.KEY_DATASET: 'tinyimagenet',
			P.KEY_DS_TRN_SET_SIZE: self.TINYIMAGENET_TRN_SET_SIZE,
			P.KEY_DS_VAL_SET_SIZE: 10000,
			P.KEY_DS_TST_SET_SIZE: 10000,
			P.KEY_DS_INPUT_SHAPE: (3, 64, 64),
			P.KEY_DS_NUM_CLASSES: 200
		}
	
	def prepare_rnd_indices(self):
		random.shuffle(self.indices)
	
	def get_train_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.TOT_TRN_SAMPLES
		return Subset(TinyImageNet(root=self.DATASET_FOLDER, split='train', transform=transform), self.indices[:num_samples])
	
	def get_val_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_VAL_SAMPLES
		if self.TRN_SET_SIZE - self.TOT_TRN_SAMPLES >= num_samples:
			# If there are enough samples left in the training set, use them for validation
			return Subset(TinyImageNet(root=self.DATASET_FOLDER, split='train', transform=transform), self.indices[self.TRN_SET_SIZE - num_samples:self.TRN_SET_SIZE])
		# Validate directly on test set otherwise
		return Subset(TinyImageNet(root=self.DATASET_FOLDER, split='val', transform=transform), range(num_samples))
	
	def get_test_split(self, transform=None, num_samples=None):
		if num_samples is None: num_samples = self.NUM_TST_SAMPLES
		return Subset(TinyImageNet(root=self.DATASET_FOLDER, split='val', transform=transform), range(num_samples))

