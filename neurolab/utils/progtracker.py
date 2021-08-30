import time

from .utils import format_time

# Class for printing progress information when iterating over datasets
class ProgressTracker:
	def __init__(self, interval, total):
		self.start_time = time.time()
		self.last_time = self.start_time
		self.interval = interval
		self.total = total
		self.last_count = 0

	def print_progress(self, count):
		curr_time = time.time()
		if curr_time - self.last_time >= self.interval or count == self.total: # Time to print progress information
			total_elapsed_time = curr_time - self.start_time
			last_elapsed_time = curr_time - self.last_time
			avg_time_per_step = total_elapsed_time / count
			inst_time_per_step = last_elapsed_time / (count - self.last_count)
			remaining_steps = self.total - count
			exp_remaining_time_str = format_time(remaining_steps * avg_time_per_step)
			elapsed_time_str = format_time(total_elapsed_time)
			inst_time_per_it_str = format_time(inst_time_per_step * self.total)
			print("\r\33[KProgress: " + str(count) + "/" + str(self.total) + " (elapsed time: " + elapsed_time_str + ", expected remaining time: " + exp_remaining_time_str + ", inst speed: " + inst_time_per_it_str + "/it)", end="\n" if count == self.total else "")
			self.last_time = curr_time
			self.last_count = count
