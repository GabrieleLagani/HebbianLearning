import neurolab.params as P


config_6l = {
	P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
	P.KEY_NET_MODULES: 'neurolab.examples.models.vision.model_6l.Net',
	P.KEY_DATA_MANAGER: 'neurolab.data.CIFAR10DataManager',
	P.KEY_TOT_TRN_SAMPLES: 40000,
	P.KEY_BATCHSIZE: 64,
	P.KEY_INPUT_SHAPE: (3, 32, 32),
	P.KEY_NUM_EPOCHS: 20,
	P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
	P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
	P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
	P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.AccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
	P.KEY_TOPKACC_K: 5,
    P.KEY_LEARNING_RATE: 1e-3,
    P.KEY_LR_DECAY: 0.5,
    P.KEY_MILESTONES: range(10, 20),
    P.KEY_MOMENTUM: 0.9,
    P.KEY_L2_PENALTY: 5e-2,
	P.KEY_DROPOUT_P: 0.5,
	P.KEY_HPMANAGER: 'neurolab.hpsearch.DiscAltMinHPManager',
	P.KEY_HPSEARCH_PARAMS: {P.KEY_LEARNING_RATE: [1e-4, 1e-3], P.KEY_L2_PENALTY: [5e-2, 5e-3]},
}

config_6l_da = {
	P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
	P.KEY_NET_MODULES: 'neurolab.examples.models.vision.model_6l.Net',
	P.KEY_DATA_MANAGER: 'neurolab.data.CIFAR10DataManager',
	P.KEY_AUGMENT_MANAGER: 'neurolab.data.CustomAugmentManager',
	P.KEY_BATCHSIZE: 64,
	P.KEY_INPUT_SHAPE: (3, 32, 32),
	P.KEY_NUM_EPOCHS: 100,
	P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
	P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
	P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
	P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.AccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
	P.KEY_TOPKACC_K: 5,
    P.KEY_LEARNING_RATE: 1e-3,
    P.KEY_LR_DECAY: 0.1,
    P.KEY_MILESTONES: [40, 70, 90],
    P.KEY_MOMENTUM: 0.9,
    P.KEY_L2_PENALTY: 3e-2,
	P.KEY_DROPOUT_P: 0.5,
	P.KEY_DA_ROT_DEGREES: 10,
}

fc_on_layer = {}
knn_on_layer = {}
for l in range(1, 6):
	fc_on_layer[str(l)] = {
		P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
		P.KEY_NET_MODULES: 'neurolab.examples.models.vision.fc.Net',
		P.KEY_NET_OUTPUTS: 'fc',
		P.KEY_DATA_MANAGER: 'neurolab.data.CIFAR10DataManager',
		P.KEY_TOT_TRN_SAMPLES: 40000,
		P.KEY_BATCHSIZE: 64,
		P.KEY_INPUT_SHAPE: (3, 32, 32),
		P.KEY_NUM_EPOCHS: 20,
		P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
		P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
		P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
		P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
		P.KEY_TOPKACC_K: [1, 5],
	    P.KEY_LEARNING_RATE: 1e-3,
	    P.KEY_LR_DECAY: 0.5,
	    P.KEY_MILESTONES: range(10, 20),
	    P.KEY_MOMENTUM: 0.9,
	    P.KEY_L2_PENALTY: 5e-4,
		P.KEY_DROPOUT_P: 0.5,
		P.KEY_PRE_NET_MODULES: ['neurolab.examples.models.vision.model_6l.Net'],
		P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/neurolab/examples/configs/vision/config_6l/iter_' + P.STR_TOKEN + '/model.pt'],
		P.KEY_PRE_NET_OUTPUTS: ['var_adaptive' + str(l)],
	}
	
	knn_on_layer[str(l)] = {
		P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
		P.KEY_NET_MODULES: 'neurolab.model.skclassif.KNNClassifier',
		P.KEY_DATA_MANAGER: 'neurolab.data.CIFAR10DataManager',
		P.KEY_TOT_TRN_SAMPLES: 40000,
		P.KEY_BATCHSIZE: 64,
		P.KEY_INPUT_SHAPE: (3, 32, 32),
		P.KEY_NUM_EPOCHS: 1,
		P.KEY_CRIT_METRIC_MANAGER: 'neurolab.optimization.metric.AccMetricManager',
		P.KEY_KNN_N_NEIGHBORS: 10,
		P.KEY_PRE_NET_MODULES: ['neurolab.examples.models.vision.model_6l.Net'],
		P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/neurolab/examples/configs/vision/config_6l/iter_' + P.STR_TOKEN + '/model.pt'],
		P.KEY_PRE_NET_OUTPUTS: ['var_adaptive' + str(l)],
	}

