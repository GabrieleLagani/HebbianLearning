from neurolab import params as P

# TODO: Add convergence to target performance value(s).
# TODO: Correct MAP metric and prepare retrieval experiments
# TODO: add saving of final model in addition to best model. Add possibility to load last model by disabling early stop
#  and to return final result instead of best result when early stop is disabled. In alternative, save only last model
#  instead of best when early stop is disabled.
# TODO: Add plots of weight norms, weight change norms, stopping condition when weights stop changing, elastic weight loss.
# TODO: Write documentation, add checks on user provided parameters, print exception in try-except-pass blocks, publish pip package.
# TODO: (as needed)
#  Add adversarial examples.
#  Add deep layer visualization.
#  Add other dataset configurations.
#  Add transfer learning configurations.
#  Implement a generic ensamble model that takes pre-trained models from configuration.
#  Implement test-time data augmentation
#  Add additional data augmentation transformations (random blur, random noise, cutout, mixup).
#  Add other tasks and datasets (miniimagenet, imagenette, emnist, kmnist, fashion-mnist, svhn, celeba, pascal voc).
# TODO: Implement dispatch of different seeds/different hyperparam configs to different gpus.
# TODO: Implement dispatch of different layers to different gpus.


stack_base = [{
	P.KEY_STACK_CONFIG: 'configs.base.config_hebb',
	P.KEY_STACK_MODE: P.MODE_TRN,
	P.KEY_STACK_DEVICE: 'cuda:0',
	P.KEY_STACK_SEEDS: [0],
	P.KEY_STACK_TOKENS: None,
	P.KEY_STACK_HPSEARCH: False,
	P.KEY_STACK_HPSEEDS: [100],
	P.KEY_STACK_DATASEEDS: [200],
	P.KEY_STACK_CHECKPOINT: None,
	P.KEY_STACK_RESTART: True,
	P.KEY_STACK_CLEARHIST: True,
	P.KEY_STACK_BRANCH: None,
}]

