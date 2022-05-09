import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .. import params as P
from .. import utils


# Data augmentation transform manager base class
class AugmentManager:
	def __init__(self, config):
		self.config = config
	
	def get_transform(self):
		raise NotImplementedError
	
	def get_transform_summary(self):
		raise NotImplementedError

# Data manager base class
class DataManager:
	
	def __init__(self, config):
		self.config = config
		
		# Prepare rng state
		self.DATASEED = P.GLB_PARAMS[P.KEY_GLB_DATASEEDS][self.config.ITER_NUM % len(P.GLB_PARAMS[P.KEY_GLB_DATASEEDS])]
		prev_rng_state = utils.get_rng_state()
		utils.set_rng_seed(self.DATASEED)
		self.rng_state = utils.get_rng_state() # Saved rng state is used during stats computation to set rng state before accessing the dataset, in order to have deterministic stats computation

		# Constants for data loading
		self.DATASET_METADATA = self.get_dataset_metadata()
		P.GLB_PARAMS[P.KEY_DATASET_METADATA] = self.DATASET_METADATA # Make dataset metadata available as global params
		self.DATASET_NAME = self.DATASET_METADATA[P.KEY_DATASET]
		self.DATASET_FOLDER = os.path.join(P.DATASETS_FOLDER, self.DATASET_NAME)
		self.TRN_SET_SIZE = self.DATASET_METADATA[P.KEY_DS_TRN_SET_SIZE]
		self.VAL_SET_SIZE = self.DATASET_METADATA[P.KEY_DS_VAL_SET_SIZE]
		self.TST_SET_SIZE = self.DATASET_METADATA[P.KEY_DS_TST_SET_SIZE]
		self.TOT_TRN_SAMPLES = self.config.CONFIG_OPTIONS.get(P.KEY_TOT_TRN_SAMPLES, self.TRN_SET_SIZE)
		self.NUM_TRN_SAMPLES = self.config.CONFIG_OPTIONS.get(P.KEY_NUM_TRN_SAMPLES, self.TOT_TRN_SAMPLES)
		self.NUM_VAL_SAMPLES = self.config.CONFIG_OPTIONS.get(P.KEY_NUM_VAL_SAMPLES, self.VAL_SET_SIZE)
		self.NUM_TST_SAMPLES = self.config.CONFIG_OPTIONS.get(P.KEY_NUM_TST_SAMPLES, self.TST_SET_SIZE)
		self.NUM_CLASSES = self.DATASET_METADATA[P.KEY_DS_NUM_CLASSES]
		self.DEFAULT_INPUT_SHAPE = self.DATASET_METADATA[P.KEY_DS_INPUT_SHAPE]
		self.INPUT_SHAPE = self.config.CONFIG_OPTIONS.get(P.KEY_INPUT_SHAPE, None)
		if self.INPUT_SHAPE is None: self.INPUT_SHAPE = self.DEFAULT_INPUT_SHAPE
		self.DEFAULT_INPUT_CHANNELS = self.DEFAULT_INPUT_SHAPE[0]
		self.INPUT_CHANNELS = self.INPUT_SHAPE[0]
		self.INPUT_SIZE = min(self.INPUT_SHAPE[1], self.INPUT_SHAPE[2])
		self.INPUT_SIZE_TOTAL = utils.shape2size(self.INPUT_SHAPE)
		self.BATCHSIZE = self.config.CONFIG_OPTIONS[P.KEY_BATCHSIZE]
		self.AUGMENT_MANAGER = self.config.CONFIG_OPTIONS.get(P.KEY_AUGMENT_MANAGER, None)
		self.augment_manager = utils.retrieve(self.AUGMENT_MANAGER)(self.config) if self.AUGMENT_MANAGER is not None else None
		self.AUGM_BEFORE_STATS = self.AUGMENT_MANAGER is not None and self.config.CONFIG_OPTIONS.get(P.KEY_AUGM_BEFORE_STATS, False)
		self.AUGM_STAT_PASSES = self.config.CONFIG_OPTIONS.get(P.KEY_AUGM_STAT_PASSES, 1) if self.AUGM_BEFORE_STATS else 1
		self.WHITEN = self.config.CONFIG_OPTIONS.get(P.KEY_WHITEN, None) # Perform whitening with smoothing const 10^-self.WHITEN
		self.NUM_WORKERS = P.GLB_PARAMS[P.KEY_GLB_NUM_WORKERS]
		self.DEVICE = P.DEVICE
		self.SYS_INFO = self.config.SYS_INFO

		# Constants related to files where dataset statistics are computed and saved
		self.STATS_FILE = os.path.join(P.STATS_FOLDER, self.DATASET_NAME, 'stats') + str(self.DATASEED) + '_' + str(self.INPUT_CHANNELS) + 'x' + str(self.INPUT_SIZE) + 'x' + str(self.INPUT_SIZE) + '_' + str(self.TOT_TRN_SAMPLES) + 'x' + str(self.AUGM_STAT_PASSES) + (('_da_' + self.AUGMENT_MANAGER + '_' + self.augment_manager.get_transform_summary()) if self.AUGM_BEFORE_STATS else '') + '.pt'
		self.PFM_FILE = os.path.join(P.STATS_FOLDER, self.DATASET_NAME, 'pfm') + str(self.DATASEED) + '_' + str(self.INPUT_CHANNELS) + 'x' + str(self.INPUT_SIZE) + 'x' + str(self.INPUT_SIZE) + '_' + str(self.TOT_TRN_SAMPLES) + 'x' + str(self.AUGM_STAT_PASSES) + (('_da_' + self.AUGMENT_MANAGER + '_' + self.augment_manager.get_transform_summary()) if self.AUGM_BEFORE_STATS else '') + '.pt'
		self.SVD_FILE = os.path.join(P.STATS_FOLDER, self.DATASET_NAME, 'svd') + str(self.DATASEED) + '_' + str(self.INPUT_CHANNELS) + 'x' + str(self.INPUT_SIZE) + 'x' + str(self.INPUT_SIZE) + '_' + str(self.TOT_TRN_SAMPLES) + 'x' + str(self.AUGM_STAT_PASSES) + (('_da_' + self.AUGMENT_MANAGER + '_' + self.augment_manager.get_transform_summary()) if self.AUGM_BEFORE_STATS else '') + '.pt'
		self.ZCA_FILE = os.path.join(P.STATS_FOLDER, self.DATASET_NAME, 'zca') + str(self.DATASEED) + '_' + 'w' + str(self.WHITEN) + '_' + str(self.INPUT_CHANNELS) + 'x' + str(self.INPUT_SIZE) + 'x' + str(self.INPUT_SIZE) + '_' + str(self.TOT_TRN_SAMPLES) + 'x' + str(self.AUGM_STAT_PASSES) + (('_da_' + self.AUGMENT_MANAGER + '_' + self.augment_manager.get_transform_summary() if self.AUGM_BEFORE_STATS else '')) + '.pt'
		self.MEAN_KEY = 'mean'
		self.STD_KEY = 'std'
		self.PFM_KEY = 'pfm'
		self.S_KEY = 'svd_S'
		self.U_KEY = 'svd_U'
		self.ZCA_KEY = 'zca'
		self.RNG_STATE_KEY = 'rng'
		self.SYS_INFO_KEY = 'sys_info'
		
		# Prepare randomly shuffled indices to be used to get dataset splits
		self.prepare_rnd_indices()
		# Since rng might have been used, update saved rng state
		self.rng_state = utils.get_rng_state()
  
		# Define transformations to be applied on the data
  
		# First we need to bring the (shortest dimension of the) image to the desired input size and the number of
		# channels to the desired number.
		T_resize = transforms.Resize(self.INPUT_SIZE) if self.INPUT_CHANNELS == self.DEFAULT_INPUT_CHANNELS else transforms.Compose([transforms.Grayscale(self.INPUT_CHANNELS), transforms.Resize(self.INPUT_SIZE)])
  
		# Add other necessary transformations
		T_other = transforms.Compose([
			# Take a central square crop of the desired input size (needed in case the image is non square)
			transforms.CenterCrop(self.INPUT_SIZE),
			# ToTensor transforms the raw img to a tensor in the form [depth, width, height].
			# Additionally, pixel values are mapped from the range [0, 255] to the range [0, 1]
			transforms.ToTensor()
		])

		# Compose everything together
		self.T = transforms.Compose([T_resize, T_other])
		
		# Add data augmentation transformations, if needed
		self.T_augm = transforms.Compose([T_resize, self.augment_manager.get_transform(), T_other]) if self.AUGMENT_MANAGER is not None else self.T

		# Append normalization transformation (not needed when whitening is performed)
		self.mean = None
		self.std = None
		self.zca = None
		self.pfm = None
		self.stats_sys_info = ""
		self.zca_sys_info = ""
		self.pfm_sys_info = ""
		if self.WHITEN is None:
			self.mean, self.std, self.stats_sys_info = self.get_stats()
			# The Normalize transform subtracts mean values from each channel and divides each channel by std dev values.
			# So we bring each channel to zero mean and unit std, i.e. from range [0, 1] to range [-1, 1]
			self.T = transforms.Compose([self.T, transforms.Normalize(self.mean, self.std)])
			self.T_augm = transforms.Compose([self.T_augm, transforms.Normalize(self.mean, self.std)])
		else:
			# Compute whitening matrix, otherwise
			self.zca, self.zca_sys_info = self.get_zca()
			self.pfm, self.pfm_sys_info = self.get_per_feature_mean()
		
		# Log preprocessing details
		self.logger = utils.Logger(self.config.LOG_PATH)
		self.logger.log("")
		self.logger.log("Preprocessing details:")
		if self.WHITEN is None:
			self.logger.log("Using stats computed as follows:\n" + self.stats_sys_info)
		else:
			self.logger.log("Using per feature mean computed as follows:\n" + self.pfm_sys_info)
			self.logger.log("Using ZCA computed as follows:\n" + self.zca_sys_info)
		self.logger.log("End of preprocessing details.")
		
		# Load datasets
		self.train_set = self.load_split(self.get_train_split(transform=self.T_augm if self.AUGMENT_MANAGER is not None else self.T, num_samples=self.NUM_TRN_SAMPLES), shuffle=True)
		self.val_set = self.load_split(self.get_val_split(transform=self.T))
		self.test_set = self.load_split(self.get_test_split(transform=self.T))
		
		# Restore previous rng state
		utils.set_rng_state(prev_rng_state)
		
	# Method for obtaining dataset metadata
	def get_dataset_metadata(self):
		raise NotImplementedError
	
	# Method for preparing randomly shuffled indices before accessing random dataset splits with the following methods defined below
	def prepare_rnd_indices(self):
		pass
	
	# Method for obtaining a custom sampler for a given dataset split.
	def get_sampler(self, split):
		return None
	
	# Methods for obtaining dataset splits
	
	def get_train_split(self, transform=None, num_samples=None):
		raise NotImplementedError
	
	def get_val_split(self, transform=None, num_samples=None):
		raise NotImplementedError
	
	def get_test_split(self, transform=None, num_samples=None):
		raise NotImplementedError
	
	def load_split(self, split, shuffle=False):
		sampler = self.get_sampler(split)
		return DataLoader(split, batch_size=self.BATCHSIZE, shuffle=shuffle if sampler is None else False, sampler=sampler, num_workers=self.NUM_WORKERS)
	
	def get_train_set(self):
		return self.train_set

	def get_val_set(self):
		return self.val_set

	def get_test_set(self):
		return self.test_set
	
	# Return dataset mean value and standard deviation per channel
	def get_stats(self):
		stats_dict = utils.load_dict(self.STATS_FILE)  # Try to load stats from file
		if stats_dict is None:  # Stats file does not exist --> Compute statistics
			print("Computing statistics for dataset \"" + self.DATASET_NAME + "\"...")
			# Set RNG state
			utils.set_rng_state(self.rng_state)
			# Get the images on which to compute the statistics
			dataset = self.load_split(self.get_train_split(transform=self.T_augm if self.AUGM_BEFORE_STATS else self.T))
			sum = torch.zeros(self.INPUT_CHANNELS, device=self.DEVICE)
			sum_sq = torch.zeros(self.INPUT_CHANNELS, device=self.DEVICE)
			count = 0
			progtracker = utils.ProgressTracker(P.PROGRESS_INTERVAL, self.AUGM_STAT_PASSES * self.TOT_TRN_SAMPLES)
			for _ in range(self.AUGM_STAT_PASSES):
				for batch in dataset:
					inputs, _ = batch
					inputs = inputs.to(self.DEVICE)
					count += inputs.size(0)
					sum += inputs.mean(dim=(2, 3)).sum(0)
					sum_sq += (inputs**2).mean(dim=(2, 3)).sum(0)
					# Print progress information roughly every P.PROGRESS_INTERVAL seconds
					progtracker.print_progress(count)
			mean = sum / count
			mean_sq = sum_sq / count
			std = (mean_sq - mean**2)**0.5
			# Save statistics
			stats_dict = {self.MEAN_KEY: mean.tolist(), self.STD_KEY: std.tolist(), self.RNG_STATE_KEY: utils.get_rng_state(), self.SYS_INFO_KEY: "Stats computation details:\n" + self.SYS_INFO + "\nEnd stats computation details."}
			utils.save_dict(stats_dict, self.STATS_FILE)
			print("Statistics computed and saved.")
		# If results were recovered from file, restore RNG state as if they were computed now
		utils.set_rng_state(stats_dict[self.RNG_STATE_KEY])
		# Return results
		return stats_dict[self.MEAN_KEY], stats_dict[self.STD_KEY], stats_dict[self.SYS_INFO_KEY]

	# Return per feature mean value
	def get_per_feature_mean(self):
		pfm_dict = utils.load_dict(self.PFM_FILE)
		if pfm_dict is None: # per feature mean file does not exist --> compute per feature mean
			print("Computing per feature mean for dataset \"" + self.DATASET_NAME + "\"...")
			# Set RNG state
			utils.set_rng_state(self.rng_state)
			# Get the images on which to compute the statistics
			dataset = self.load_split(self.get_train_split(transform=self.T_augm if self.AUGM_BEFORE_STATS else self.T))
			sum = torch.zeros(self.INPUT_SIZE_TOTAL, device=self.DEVICE)
			count = 0
			progtracker = utils.ProgressTracker(P.PROGRESS_INTERVAL, self.AUGM_STAT_PASSES * self.TOT_TRN_SAMPLES)
			for _ in range(self.AUGM_STAT_PASSES):
				for batch in dataset:
					inputs, _ = batch
					inputs = inputs.to(self.DEVICE)
					sum += (inputs.view(-1, self.INPUT_SIZE_TOTAL)).sum(0)
					count += inputs.size(0)
					# Print progress information roughly every P.PROGRESS_INTERVAL seconds
					progtracker.print_progress(count)
			pfm = sum / count
			# Save per feature mean
			pfm_dict = {self.PFM_KEY: pfm, self.RNG_STATE_KEY: utils.get_rng_state(), self.SYS_INFO_KEY: "Per feature mean computation details:\n" + self.SYS_INFO + "\nEnd per feature mean computation details."}
			utils.save_dict(pfm_dict, self.PFM_FILE)
			print("Per feature mean computed and saved.")
		# If results were recovered from file, restore RNG state as if they were computed now
		utils.set_rng_state(pfm_dict[self.RNG_STATE_KEY])
		# Return results
		return pfm_dict[self.PFM_KEY].to(self.DEVICE), pfm_dict[self.SYS_INFO_KEY]

	# Return dataset covariance matrix singular value decomposition
	def get_svd(self):
		svd_dict = utils.load_dict(self.SVD_FILE)  # Try to load svd from file
		if svd_dict is None:  # svd file does not exist --> Compute covariance matrix and perform svd
			pfm, pfm_sys_info = self.get_per_feature_mean()
			print("Computing covariance matrix for dataset \"" + self.DATASET_NAME + "\"...")
			# Set RNG state
			utils.set_rng_state(self.rng_state)
			# Get the images on which to compute the statistics
			dataset = self.load_split(self.get_train_split(transform=self.T_augm if self.AUGM_BEFORE_STATS else self.T))
			cov = torch.zeros((self.INPUT_SIZE_TOTAL, self.INPUT_SIZE_TOTAL), device=self.DEVICE)
			count = 0
			progtracker = utils.ProgressTracker(P.PROGRESS_INTERVAL, self.AUGM_STAT_PASSES * self.TOT_TRN_SAMPLES)
			for _ in range(self.AUGM_STAT_PASSES):
				for batch in dataset:
					inputs, _ = batch
					inputs = inputs.to(self.DEVICE)
					inputs = inputs.view(-1, self.INPUT_SIZE_TOTAL) - pfm.view(1, self.INPUT_SIZE_TOTAL)
					cov += inputs.t().matmul(inputs)
					count += inputs.size(0)
					# Print progress information roughly every P.PROGRESS_INTERVAL seconds
					progtracker.print_progress(count)
			cov = cov / (count - 1)
			print("Computing SVD of covariance matrix for dataset \"" + self.DATASET_NAME + "\" (this might take a while)...")
			U, S, V = torch.svd(cov)
			# Save covariance matrix and svd
			svd_dict = {self.S_KEY: S, self.U_KEY: U, self.RNG_STATE_KEY: utils.get_rng_state(), self.SYS_INFO_KEY: "SVD computation details:\n" + self.SYS_INFO + "\nUsing per feature mean computed as follows:\n" + pfm_sys_info + "\nEnd SVD computation details."}
			utils.save_dict(svd_dict, self.SVD_FILE)
			print("SVD computed and saved.")
		# If results were recovered from file, restore RNG state as if they were computed now
		utils.set_rng_state(svd_dict[self.RNG_STATE_KEY])
		# Return results
		return svd_dict[self.S_KEY].to(self.DEVICE), svd_dict[self.U_KEY].to(self.DEVICE), svd_dict[self.SYS_INFO_KEY]

	# Return dataset ZCA matrix
	def get_zca(self):
		SMOOTHING_CONST = 10**(-self.WHITEN)
		zca_dict = utils.load_dict(self.ZCA_FILE)  # Try to load zca from file
		if zca_dict is None:  # zca file does not exist --> Compute zca matrix
			S, U, svd_sys_info = self.get_svd()
			print("Computing ZCA matrix for dataset \"" + self.DATASET_NAME + "\"...")
			zca = U.matmul(torch.diag((S + SMOOTHING_CONST) ** -0.5).matmul(U.t()))
			# Save zca matrix
			zca_dict = {self.ZCA_KEY: zca, self.SYS_INFO_KEY: "ZCA computation details:\n" + self.SYS_INFO + "\nUsing SVD computed as follows:\n" + svd_sys_info + "\nEnd ZCA computation details."}
			utils.save_dict(zca_dict, self.ZCA_FILE)
			print("ZCA matrix computed and saved.")
		return zca_dict[self.ZCA_KEY].to(self.DEVICE), zca_dict[self.SYS_INFO_KEY]

	# Method for preprocessing a batch of data
	def preprocess(self, x):
		# Preprocessing consists in whitening the data samples
		if self.WHITEN is not None: return torch.matmul(self.zca, (x.view(-1, self.INPUT_SIZE_TOTAL) - self.pfm.view(1, self.INPUT_SIZE_TOTAL)).t()).t().view(-1, *self.INPUT_SHAPE)
		return x
