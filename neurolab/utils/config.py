import os

from .. import params as P
from . import utils

class Config:
	def __init__(self, config_id, config_options, mode, iter_num, iter_id, result_base_folder, tokens, summary=None):
		self.CONFIG_ID = config_id
		self.CONFIG_OPTIONS = config_options
		self.MODE = mode
		self.ITER_NUM = iter_num
		self.ITER_ID = iter_id
		self.SUMMARY = summary
		
		# Prepare configuration options for the experiment
		if type(self.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MODULES, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_PRE_NET_MODULES] = [self.CONFIG_OPTIONS[P.KEY_PRE_NET_MODULES]]
		if type(self.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MDL_PATHS, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS] = [self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS]]
		if type(self.CONFIG_OPTIONS.get(P.KEY_PRE_NET_OUTPUTS, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_PRE_NET_OUTPUTS] = [self.CONFIG_OPTIONS[P.KEY_PRE_NET_OUTPUTS]]
		if self.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MDL_PATHS, None) is not None and tokens is not None:
			splt_tokens = tokens.split(',')
			for i in range(len(self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS])):
				self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS][i] = self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS][i].replace(P.STR_TOKEN, splt_tokens[i])
		if type(self.CONFIG_OPTIONS.get(P.KEY_NET_MODULES, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_NET_MODULES] = [self.CONFIG_OPTIONS[P.KEY_NET_MODULES]]
		if type(self.CONFIG_OPTIONS.get(P.KEY_NET_MDL_PATHS, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_NET_MDL_PATHS] = [self.CONFIG_OPTIONS[P.KEY_NET_MDL_PATHS]]
		if type(self.CONFIG_OPTIONS.get(P.KEY_NET_OUTPUTS, None)) is str:
			self.CONFIG_OPTIONS[P.KEY_NET_OUTPUTS] = [self.CONFIG_OPTIONS[P.KEY_NET_OUTPUTS]]
		if self.CONFIG_OPTIONS.get(P.KEY_NET_MDL_PATHS, None) is not None and tokens is not None:
			splt_tokens = tokens.split(',')
			for i in range(len(self.CONFIG_OPTIONS[P.KEY_NET_MDL_PATHS])):
				self.CONFIG_OPTIONS[P.KEY_NET_MDL_PATHS][i] = self.CONFIG_OPTIONS[P.KEY_NET_MDL_PATHS][i].replace(P.STR_TOKEN, splt_tokens[i + (len(self.CONFIG_OPTIONS[P.KEY_PRE_NET_MDL_PATHS]) if self.CONFIG_OPTIONS.get(P.KEY_PRE_NET_MDL_PATHS, None) is not None else 0)])
		if type(self.CONFIG_OPTIONS.get(P.KEY_CRIT_METRIC_MANAGER, None)) is str: self.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER] = [self.CONFIG_OPTIONS[P.KEY_CRIT_METRIC_MANAGER]]
		
		# Constrct string containing config information
		self.CONFIG_INFO = ""
		self.CONFIG_INFO += "CONFIG_ID: " + self.CONFIG_ID + "\n"
		self.CONFIG_INFO += "CONFIG_OPTIONS: " + str(self.CONFIG_OPTIONS) + "\n"
		self.CONFIG_INFO += "MODE: " + str(self.MODE) + "\n"
		self.CONFIG_INFO += "ITER_NUM: " + str(self.ITER_NUM) + "\n"
		self.CONFIG_INFO += "ITER_ID: " + str(self.ITER_ID) + "\n"

		# Path to the folder where results are saved
		self.RESULT_BASE_FOLDER = result_base_folder
		# Path to the folder where the results for the specific iteration are saved
		self.RESULT_FOLDER = os.path.join(self.RESULT_BASE_FOLDER, 'iter' + str(self.ITER_ID))
		# Path where to save checkpoints
		self.CHECKPOINT_FOLDER = os.path.join(self.RESULT_FOLDER, 'checkpoints')
		# Path where to save figures
		self.FIGURE_FOLDER = os.path.join(self.RESULT_FOLDER, 'figures')
		# Path where to save logs
		self.LOG_FOLDER = os.path.join(self.RESULT_FOLDER, 'logs')
		self.LOG_PATH = os.path.join(self.LOG_FOLDER, 'trn_log.txt' if self.MODE == P.MODE_TRN else 'tst_log.txt')
		# Path where to save the models
		self.SAVED_MDL_FOLDER = os.path.join(self.RESULT_FOLDER, 'models')
		self.SAVED_MDL_PATHS = [os.path.join(self.SAVED_MDL_FOLDER, 'model' + str(i) + '.pt') for i in range(len(self.CONFIG_OPTIONS[P.KEY_NET_MODULES]))] if self.CONFIG_OPTIONS.get(P.KEY_NET_MODULES, None) is not None else []
		
		# Get system information
		self.SYS_INFO = utils.get_sys_info()
		
		
		

