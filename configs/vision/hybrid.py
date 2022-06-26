from neurolab import params as P
import params as PP
from .meta import *


gdes_on_hebb_layer = {}
hebb_on_gdes_layer = {}
ghg = {}


for ds in datasets:
	for da in da_strategies:
		for lrn_rule in lrn_rules:
			for l in range(1, num_layers[ds] - 2):
				gdes_on_hebb_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.gdes.top_' + str(num_layers[ds]) + 'l.top' + str(l) + '.Net',
					P.KEY_NET_OUTPUTS: net_outputs[ds],
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None if lrn_rule_keys[lrn_rule] != 'hwta' else 2,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: da_mult[da] * 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
					P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
				    P.KEY_LEARNING_RATE: 1e-3,
				    P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
		            P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
				    P.KEY_MOMENTUM: 0.9,
				    P.KEY_L2_PENALTY: l2_penalties[ds],
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.hebb.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/hebb/config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}
				
				hebb_on_gdes_layer[str(l) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.top_' + str(num_layers[ds]) + 'l.top' + str(l) + '.Net',
					P.KEY_NET_OUTPUTS: net_outputs[ds],
					P.KEY_DATA_MANAGER: data_managers[ds],
					P.KEY_AUGMENT_MANAGER: da_managers[da],
					P.KEY_AUGM_BEFORE_STATS: True,
					P.KEY_AUGM_STAT_PASSES: da_mult[da],
					P.KEY_WHITEN: None,
					P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
					P.KEY_BATCHSIZE: batch_sizes[ds],
					P.KEY_INPUT_SHAPE: input_shapes[ds],
					P.KEY_NUM_EPOCHS: 20,
					P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
					P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
					P.KEY_TOPKACC_K: [1, 5],
				    P.KEY_LEARNING_RATE: hebb_lrn_rates[lrn_rule_keys[lrn_rule]][ds],
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
					P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net'],
					P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'],
					P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l)],
				}
			
			for l1 in range(1, num_layers[ds] - 1):
				for l2 in range(l1 + 1, num_layers[ds]):
					ghg[str(l1) + '_' + str(l2) + '_' + lrn_rule + '_' + ds + da_names[da]] = {
						P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
						P.KEY_NET_MODULES: 'models.gdes.fc.Net' if l2 == num_layers[ds] - 1 else ('models.gdes.fc2.Net' if l2 == num_layers[ds] - 2 else ('models.gdes.top_' + str(num_layers[ds]) + 'l.top' + str(l2) + '.Net')),
						P.KEY_NET_OUTPUTS: 'fc' if l2 == num_layers[ds] - 1 else ('fc2' if l2 == num_layers[ds] - 2 else net_outputs[ds]),
						P.KEY_DATA_MANAGER: data_managers[ds],
						P.KEY_AUGMENT_MANAGER: da_managers[da],
						P.KEY_AUGM_BEFORE_STATS: True,
						P.KEY_AUGM_STAT_PASSES: da_mult[da],
						P.KEY_WHITEN: None,
						P.KEY_TOT_TRN_SAMPLES: tot_trn_samples[ds],
						P.KEY_BATCHSIZE: batch_sizes[ds],
						P.KEY_INPUT_SHAPE: input_shapes[ds],
						P.KEY_NUM_EPOCHS: da_mult[da] * 20,
						P.KEY_OPTIM_MANAGER: 'neurolab.optimization.optim.SGDOptimManager',
						P.KEY_SCHED_MANAGER: 'neurolab.optimization.sched.MultiStepSchedManager',
						P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntMetricManager',
						P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
						P.KEY_TOPKACC_K: [1, 5],
					    P.KEY_LEARNING_RATE: 1e-3,
					    P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
		                P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
					    P.KEY_MOMENTUM: 0.9,
					    P.KEY_L2_PENALTY: 5e-4 if l2 > num_layers[ds] - 3 else l2_penalties[ds],
						P.KEY_DROPOUT_P: 0.5,
						P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
						PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
						PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
						P.KEY_PRE_NET_MODULES: ['models.gdes.model_' + str(num_layers[ds]) + 'l.Net', 'models.hebb.fc2.Net' if l1 == num_layers[ds] - 2 else ('models.hebb.top_' + str(num_layers[ds]) + 'l.top' + str(l1) + '.Net')],
						P.KEY_PRE_NET_MDL_PATHS: [P.PROJECT_ROOT + '/results/configs/vision/gdes/config_base_gdes[' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt',
						                          P.PROJECT_ROOT + (('/results/configs/vision/gdes/hebb_fc2_on_gdes_layer[' + str(num_layers[ds] - 2) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt') if l1 == num_layers[ds] - 2 else 
						                                            ('/results/configs/vision/hybrid/hebb_on_gdes_layer[' + str(l1) + '_' + lrn_rule + '_' + ds + da_names[da] + ']/iter' + P.STR_TOKEN + '/models/model0.pt'))],
						P.KEY_PRE_NET_OUTPUTS: ['bn' + str(l1), 'bn1' if l1 == num_layers[ds] - 2 else 'bn' + str(l2)],
					}

