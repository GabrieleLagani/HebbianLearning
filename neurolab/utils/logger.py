import time
from datetime import datetime
import os


# Class for printing on screen and on log file
class Logger:
	def __init__(self, log_file_path):
		self.log_file = log_file_path
		os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
	
	def log(self, msg):
		log_file = open(self.log_file, 'a')
		log_file.write("[" + str(datetime.utcfromtimestamp(time.time())) + "] " + str(msg) + '\n')
		log_file.flush()
		log_file.close()

	def print_and_log(self, msg):
		self.log(msg)
		print(msg)
	
	def clear(self):
		os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
		log_file = open(self.log_file, 'w')
		log_file.write('')
		log_file.flush()
		log_file.close()

