import os, subprocess, platform
from importlib import import_module
import requests
import time
import csv
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.stats as st
import numpy as np
import torch

from .. import params as P


# Return formatted string with time information
def format_time(seconds):
	seconds = int(seconds)
	minutes, seconds = divmod(seconds, 60)
	hours, minutes = divmod(minutes, 60)
	return str(hours) + "h " + str(minutes) + "m " + str(seconds) + "s"

# Convert tensor shape to total tensor size
def shape2size(shape):
	size = 1
	for s in shape: size *= s
	return size

# Convert (dictionary of) tensors to (dictionary of) corresponding shapes
def tens2shape(input):
	return {k: tuple(input[k].size())[1:] if isinstance(input[k], torch.Tensor) else input[k] for k in input.keys()} if isinstance(input, dict) else tuple(input.size())[1:]

# Convert dense-encoded vector to one-hot encoded
def dense2onehot(tensor, n):
	return torch.zeros(tensor.size(0), n, device=tensor.device).scatter_(1, tensor.unsqueeze(1).long(), 1)

# Checks whether curr_res is better than best_res according whether the evaluation is HB or LB
def is_better(curr_res, best_res, hb):
	if best_res is None: return True
	return (curr_res > best_res) if hb else (curr_res < best_res)

# Checks whether curr_res is within perc% of best_res or higher/lower according whether the evaluation is HB or LB
def is_converged(curr_res, best_res, perc, hb):
	if best_res is None: return True
	return (curr_res >= (1 - perc) * best_res) if hb else (curr_res <= (1 + perc) * best_res)

# Retrieve a custom module or object provided by the user by full name in dot notation as string. If the object is a
# dictionary, it is possible to retrieve a specific element of the dictionary with the square bracket indexing notation.
# NB: dictionary keys must always be strings.
def retrieve(name):
	if '[' in name:
		name, key = name.split('[', 1)
		key = key.rsplit(']', 1)[0]
		prefix, suffix = name.rsplit('.', 1)
		return getattr(import_module(prefix), suffix)[key]
	
	prefix, suffix = name.rsplit('.', 1)
	return getattr(import_module(prefix), suffix)

# Set rng seed
def set_rng_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

# Set rng state
def set_rng_state(state):
	random.setstate(state['python_rng'])
	np.random.set_state(state['numpy_rng'])
	torch.set_rng_state(state['pytorch_rng'])
	torch.cuda.set_rng_state_all(state['pytorch_rng_cuda'])

# Get rng state
def get_rng_state():
	state = {}
	state['python_rng'] = random.getstate()
	state['numpy_rng'] = np.random.get_state()
	state['pytorch_rng'] = torch.get_rng_state()
	state['pytorch_rng_cuda'] = torch.cuda.get_rng_state_all()
	return state

# Save a dictionary (e.g. representing a trained model) in the specified path
def save_dict(d, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	torch.save(d, path)

# Load a dictionary (e.g. representing a trained model) from the specified path
def load_dict(path):
	d = None
	try: d = torch.load(path, map_location='cpu')
	except: pass
	return d

# Produces a dictionary from an object. If the object has a state_dict method, returns the result of state_dict, otherwise
# returns a dictionary containing the object itself, ready to be serialized. NB: do not call utils.obj2dict(self) inside
# the state_dict method of your objects. Use utils.state_dict(self) instead.
def obj2dict(obj):
	try:
		return obj.state_dict()
	except:
		return {'obj': obj} # Should I call utils.state_dict(obj) recursively if attribute has __dict__? No, so that the user can choose to save object by serialization by not providing obj.state_dict() method.

# Loads a dictionary into an object. If the object has a load_state_dict method, this method is used to load the dictionary
# into the object, and the resulting object is returned. Otherwise, we expect the dictionary to contain the object itself
# with key 'obj', and this dictionary entry is returned. NB: do not call utils.dict2obj(self) inside the load_state_dict
# method of your objects. Use utils.load_state_dict(self) instead.
def dict2obj(d, obj=None):
	try:
		obj.load_state_dict(d)
		return obj
	except:
		return d['obj']

# Helper function to implement state_dict method of some objects. Returns the state of the given object as a dictionary.
# This is obtained by converting each attribute of the object to a dictionary element.
def state_dict(obj):
	d = obj.__dict__.copy()
	for k in d: d[k] = obj2dict(getattr(obj, k))
	return d

# Helper function to implement state_dict method of some objects. Sets the state of the given object from the dictionary.
# This is obtained by setting an object attribute, for each dictionary key, to the corresponding dictionary element.
def load_state_dict(obj, d):
	for k in d:
		if hasattr(obj, k):
			setattr(obj, k, dict2obj(d[k], getattr(obj, k)))

# Return list of checkpoints in a given folder
def get_checkpoint_list(checkpoint_folder):
	return [int(f.split('checkpoint')[1].split('.pt')[0])
			for f in os.listdir(checkpoint_folder)
			if f.startswith('checkpoint') and f.endswith('.pt')
			and f.split('checkpoint')[1].split('.pt')[0].isdigit()]

# Remove checkpoint files older than latest_checkpoint from checkpoint_folder
def clear_checkpoints(checkpoint_folder, latest_checkpoint, clearhist):
	for c in get_checkpoint_list(checkpoint_folder):
		if c > latest_checkpoint or (c < latest_checkpoint and clearhist):
			os.remove(os.path.join(checkpoint_folder, 'checkpoint' + str(c) + '.pt'))

# Save a figure showing train and validation results in the specified file
def save_trn_curve_plot(train_result_data, val_result_data, path, label='result'):
	graph = plt.axes(xlabel='epoch', ylabel=label)
	graph.plot(list(train_result_data.keys()), list(train_result_data.values()), label='train')
	graph.plot(list(val_result_data.keys()), list(val_result_data.values()), label='val.')
	graph.grid(True)
	graph.legend()
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig = graph.get_figure()
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Function to print a grid of images (e.g. representing learned kernels)
def plot_grid(tensor, path, num_rows=8, num_cols=12, bounds=None, norm_sigm=False):
	if bounds is None: bounds = (tensor.min(), tensor.max())
	if not norm_sigm: # Max-min normalization
		tensor = ((tensor - bounds[0])/(bounds[1] - bounds[0]))
	else: # Sigmoidal normalization
		tensor = torch.sigmoid((tensor - tensor.mean())/tensor.std())
	tensor = tensor.permute(0, 2, 3, 1).cpu().detach().numpy()

	fig = plt.figure()
	for i in range(min(tensor.shape[0], num_rows * num_cols)):
		ax1 = fig.add_subplot(num_rows,num_cols,i+1)
		ax1.imshow(tensor[i])
		ax1.axis('off')
		ax1.set_xticklabels([])
		ax1.set_yticklabels([])
	plt.subplots_adjust(wspace=0.1, hspace=0.1)
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path, bbox_inches='tight')
	plt.close(fig)

# Add an entry containing the seed of a training iteration and the test accuracy of the corresponding model to a csv file
def update_csv(iter_id, result, path, ci_levels=(0.9, 0.95, 0.98, 0.99, 0.995)):
	AVG_KEY = 'AVG'
	CI_KEYS = {ci_lvl: str(ci_lvl*100) + "% CI" for ci_lvl in ci_levels}
	HEADER = ('ITER_ID', 'RESULT')
	d = {}
	try:
		with open(path, 'r') as csv_file:
			reader = csv.reader(csv_file)
			d = dict(reader)
			d.pop(HEADER[0], None)
			d.pop(AVG_KEY, None)
			for ci_lvl in ci_levels: d.pop(CI_KEYS[ci_lvl], None)
	except: pass
	d[str(iter_id)] = str(result)
	with open(path, mode='w', newline='') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(HEADER)
		for k, v in d.items(): writer.writerow([k, v])
		if len(d) > 1:
			values = list(map(float, d.values()))
			avg = sum(values)/len(values)
			se = st.sem(values)
			writer.writerow([AVG_KEY, str(avg)])
			for ci_lvl in ci_levels:
				ci = st.t.interval(ci_lvl, len(values) - 1, loc=avg, scale=se)
				ci_str = "+/- " + str((ci[1] - ci[0])/2)
				writer.writerow([CI_KEYS[ci_lvl], ci_str])

# Download resource from Google Drive
def download_large_file_from_drive(id, dest, print_interval=2):
	URL = "https://docs.google.com/uc?export=download"
	CHUNK_SIZE = 32768
	# Start a first request session to get a token
	session = requests.Session()
	response = session.get(URL, params={'id': id}, stream=True)
	
	# Get confirm token
	token = None
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			token = value
			break
	# Start a second request session to get the actual resource
	if token is not None:
		params = {'id': id,'confirm': token}
		response = session.get(URL, params=params, stream=True)
	# Save resource to disk
	with open(dest, 'wb') as f:
		total_length = response.headers.get('Content-Length')
		if total_length is not None: total_length = int(total_length)
		downloaded = 0
		start_time = time.time()
		last_time = start_time
		last_downloaded = downloaded
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk is not None:
				f.write(chunk) # filter out keep-alive new chunks
				downloaded += len(chunk)
				curr_time = time.time()
				if (curr_time - last_time >= print_interval) or (total_length is not None and downloaded >= total_length):
					elapsed_time = curr_time - start_time
					avg_speed = downloaded/elapsed_time
					inst_speed = (downloaded - last_downloaded)/(curr_time - last_time)
					exp_remaining_time = (avg_speed * (total_length - downloaded)) if total_length is not None else None
					elapsed_time_str = format_time(elapsed_time)
					exp_remaining_time_str = format_time(exp_remaining_time) if exp_remaining_time is not None else "?"
					print("\r\33[KDownloaded: {}% ({}/{} KB, elaplsed time: {}, expected remaining time: {}, inst speed: {} KB/s)".format((100 * downloaded/total_length) if total_length is not None else "?", downloaded/1000, (total_length/1000) if total_length is not None else "?", elapsed_time_str, exp_remaining_time_str, inst_speed/1000), end="")
					last_time = curr_time
					last_downloaded = downloaded
		print("")

# Method for obtaining system information
def get_sys_info():
	# Get CPU name and RAM
	system = platform.system()
	cpu = ""
	cpu_ram = ""
	if system == "Windows":
		cpu = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode("utf-8").strip().split("\n")[1]
		cpu_ram = "{:.2f}".format(int(
			subprocess.check_output(["wmic", "computersystem", "get", "totalphysicalmemory"]).decode("utf-8").strip().split("\n")[1]) / (1 << 30)) + " GiB"
	elif system == "Darwin":
		cpu = subprocess.check_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
		cpu_ram = "{:.2f}".format(int(
			subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"]).decode("utf-8").strip()) / (1 << 30)) + " GiB"
	elif system == "Linux":
		all_info = subprocess.check_output(["cat", "/proc/cpuinfo"]).decode("utf-8").strip()
		for line in all_info.split("\n"):
			if line.startswith("model name"):
				cpu = line.split(": ")[1]
				break
		all_info = subprocess.check_output(["cat", "/proc/meminfo"]).decode("utf-8").strip()
		for line in all_info.split("\n"):
			if line.startswith("MemTotal"):
				cpu_ram = "{:.2f}".format(int(line.split(": ")[1].strip(" kB")) / (1 << 20)) + " GiB"
				break

	# Get GPU name and RAM
	gpu = ""
	gpu_ram = ""
	cuda = ""
	cudnn = ""
	if P.DEVICE != 'cpu':
		gpu = torch.cuda.get_device_name(P.DEVICE)
		gpu_ram = "{:.2f}".format(torch.cuda.get_device_properties(P.DEVICE).total_memory / (1 << 30)) + " GiB"
		cuda = torch.version.cuda
		cudnn = str(torch.backends.cudnn.version())

	# Check if running on Google Colab
	in_colab = True
	try: import google.colab
	except:	in_colab = False

	# Construct string containing system information
	SYS_INFO = ""
	SYS_INFO += "CPU: " + cpu + "\n"
	SYS_INFO += "CPU_RAM: " + cpu_ram + "\n"
	SYS_INFO += "DEVICE: " + P.DEVICE + "\n"
	SYS_INFO += "GPU: " + gpu + "\n"
	SYS_INFO += "GPU_RAM: " + gpu_ram + "\n"
	SYS_INFO += "CUDA: " + cuda + "\n"
	SYS_INFO += "CUDNN: " + cudnn + "\n"
	SYS_INFO += "OS: " + platform.platform() + "\n"
	SYS_INFO += "IN_COLAB: " + str(in_colab) + "\n"
	SYS_INFO += "PYTHON_VERSION: " + platform.python_version() + "\n"
	SYS_INFO += "PACKAGE_VERSIONS: " + str(P.ADDITIONAL_INFO) + "\n"
	SYS_INFO += "GLB_PARAMS: " + str(P.GLB_PARAMS)
	
	return SYS_INFO

