import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .. import params as P
from .data import AugmentManager


# Custom transform for random input resize
class RandomResize:
	def __init__(self, min_size, max_size):
		self.min_size = min_size
		self.max_size = max_size

	def __call__(self, x):
		return TF.resize(x, random.randint(self.min_size, self.max_size))

# Custom data augmentation transform manager
class CustomAugmentManager(AugmentManager):
	def __init__(self, config):
		super().__init__(config)
		
		input_size = config.CONFIG_OPTIONS.get(P.KEY_INPUT_SHAPE, P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_INPUT_SHAPE])[1]
		rel_delta = config.CONFIG_OPTIONS.get(P.KEY_DA_REL_DELTA, 0.25)
		delta = int(rel_delta * input_size)
		
		jit_brightness = config.CONFIG_OPTIONS.get(P.KEY_DA_JIT_BRIGHTNESS, 0.1)
		jit_contrast = config.CONFIG_OPTIONS.get(P.KEY_DA_JIT_CONTRAST, 0.1)
		jit_saturation = config.CONFIG_OPTIONS.get(P.KEY_DA_JIT_SATURATION, 0.1)
		jit_hue = config.CONFIG_OPTIONS.get(P.KEY_DA_JIT_HUE, 20 / 360)
		jit_p = config.CONFIG_OPTIONS.get(P.KEY_DA_JIT_P, 0.5)
		grayscale_p = config.CONFIG_OPTIONS.get(P.KEY_DA_GREYSCALE_P, 0.2)
		persp_scale = config.CONFIG_OPTIONS.get(P.KEY_DA_PERSP_SCALE, 0.25)
		persp_p = config.CONFIG_OPTIONS.get(P.KEY_DA_PERSP_P, 0.3)
		resize_p = config.CONFIG_OPTIONS.get(P.KEY_DA_RESIZE_P, 0.3)
		rot_degrees = config.CONFIG_OPTIONS.get(P.KEY_DA_ROT_DEGREES, 180)
		rot_p = config.CONFIG_OPTIONS.get(P.KEY_DA_ROT_P, 0.3)
		transl_p = config.CONFIG_OPTIONS.get(P.KEY_DA_TRANSL_P, 0.5)
		
		# A textual summary of the transformation
		self.TRANSFORM_SUMMARY = "delta" + str(delta) + "jit_brightness" + str(jit_brightness) + "jit_contrast" + str(jit_contrast) + \
			"jit_saturation" + str(jit_saturation) + "jit_hue" + str(jit_hue) + "jit_p" + str(jit_p) + "grayscale_p" + str(grayscale_p) + \
			"persp_scale" + str(persp_scale) + "persp_p" + str(persp_p) + "resize_p" + str(resize_p) + \
			"rot_degrees" + str(rot_degrees) + "rot_p" + str(rot_p) + "transl_p" + str(transl_p)
		
		T_augm = []
		# Random blur
		# Random noise
		T_augm.append(transforms.RandomApply([
			transforms.ColorJitter(brightness=jit_brightness, contrast=jit_contrast, saturation=jit_saturation, hue=jit_hue)
		], p=jit_p))
		T_augm.append(transforms.RandomGrayscale(p=grayscale_p))
		T_augm.append(transforms.RandomHorizontalFlip())
		T_augm.append(transforms.RandomApply([
			transforms.Resize(input_size + delta // 2), # Random perspective tends to shrink the image, so we enlarge it beforehand
			transforms.RandomPerspective(distortion_scale=persp_scale, p=1.0)
		], p=persp_p))
		T_augm.append(transforms.RandomApply([
			transforms.Lambda(RandomResize(input_size - delta, input_size + delta))  # Random rescale
		], p=resize_p))
		T_augm.append(transforms.RandomApply([
			transforms.RandomRotation(degrees=rot_degrees, expand=True)
		], p=rot_p))
		# Random occlusion
		T_augm.append(transforms.RandomApply([
			transforms.CenterCrop(input_size + delta),  # Take a fixed-size central crop
			transforms.RandomCrop(input_size) # Take a smaller fixed-size crop at random position (random translation)
		], p=transl_p))
		self.T_augm = transforms.Compose(T_augm)
		
	def get_transform(self):
		return self.T_augm
	
	def get_transform_summary(self):
		return self.TRANSFORM_SUMMARY

# Custom data augmentation transform manager
class LightCustomAugmentManager(AugmentManager):
	def __init__(self, config):
		super().__init__(config)
		
		input_size = config.CONFIG_OPTIONS.get(P.KEY_INPUT_SHAPE, P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_INPUT_SHAPE])[1]
		rel_delta = config.CONFIG_OPTIONS.get(P.KEY_DA_REL_DELTA, 0.25)
		delta = int(rel_delta * input_size)
		
		resize_p = config.CONFIG_OPTIONS.get(P.KEY_DA_RESIZE_P, 0.3)
		rot_degrees = config.CONFIG_OPTIONS.get(P.KEY_DA_ROT_DEGREES, 10)
		rot_p = config.CONFIG_OPTIONS.get(P.KEY_DA_ROT_P, 0.3)
		transl_p = config.CONFIG_OPTIONS.get(P.KEY_DA_TRANSL_P, 0.5)
		
		# A textual summary of the transformation
		self.TRANSFORM_SUMMARY = "delta" + str(delta) + "resize_p" + str(resize_p) + \
			"rot_degrees" + str(rot_degrees) + "rot_p" + str(rot_p) + "transl_p" + str(transl_p)
		
		T_augm = []
		T_augm.append(transforms.RandomHorizontalFlip())
		T_augm.append(transforms.RandomApply([
			transforms.Lambda(RandomResize(input_size - delta, input_size + delta))  # Random rescale
		], p=resize_p))
		T_augm.append(transforms.RandomApply([
			transforms.RandomRotation(degrees=rot_degrees, expand=True)
		], p=rot_p))
		T_augm.append(transforms.RandomApply([
			transforms.CenterCrop(input_size + delta),  # Take a fixed-size central crop
			transforms.RandomCrop(input_size) # Take a smaller fixed-size crop at random position (random translation)
		], p=transl_p))
		self.T_augm = transforms.Compose(T_augm)
		
	def get_transform(self):
		return self.T_augm
	
	def get_transform_summary(self):
		return self.TRANSFORM_SUMMARY

