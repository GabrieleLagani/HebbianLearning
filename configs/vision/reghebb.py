from neurolab import params as P
import params as PP
from .meta import *


config_hebbreg = {}
prec_on_hebbreg = {}
config_vaereg = {}
prec_on_vaereg = {}

for ds in datasets:
	for da in da_strategies:
		for lrn_rule in lrn_rules:
			for a in hebbreg_coeffs:
				config_hebbreg[lrn_rule + '_a' + str(a) + '_'  + ds + da_names[da]] = {
					P.KEY_EXPERIMENT: 'neurolab.experiment.VisionExperiment',
					P.KEY_NET_MODULES: 'models.hebb.model_' + str(num_layers[ds]) + 'l.Net',
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
					P.KEY_ALPHA_L: a, P.KEY_ALPHA_G: 1,
					P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
				    P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
				    P.KEY_MOMENTUM: 0.9,
				    P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],
					P.KEY_DROPOUT_P: 0.5,
					P.KEY_LOCAL_LRN_RULE: lrn_rule_keys[lrn_rule],
					PP.KEY_COMPETITIVE_ACT: lrn_rule_competitive_act[lrn_rule],
					PP.KEY_COMPETITIVE_K: lrn_rule_k[lrn_rule],
					P.KEY_DEEP_TEACHER_SIGNAL: lrn_rule_dts[lrn_rule],
				}
		
		for a in hebbreg_coeffs:
			config_vaereg['a' + str(a) + '_'  + ds + da_names[da]] = {
				P.KEY_EXPERIMENT: 'neurolab.experiment.AEVisionExperiment',
				P.KEY_NET_MODULES: 'models.gdes.vae_' + str(num_layers[ds]) + 'l.Net',
				P.KEY_NET_OUTPUTS: 'vae_output',
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
				P.KEY_LOSS_METRIC_MANAGER: 'neurolab.optimization.metric.CrossEntELBOMetricManager',
				P.KEY_CRIT_METRIC_MANAGER: ['neurolab.optimization.metric.TopKAccMetricManager', 'neurolab.optimization.metric.TopKAccMetricManager'],
				P.KEY_TOPKACC_K: [1, 5],
			    P.KEY_LEARNING_RATE: 1e-3,
				P.KEY_ALPHA_U: a, P.KEY_ALPHA_S: 1,
				P.KEY_ELBO_BETA: 0.5,
				P.KEY_LR_DECAY: 0.5 if da == 'no_da' else 0.1,
			    P.KEY_MILESTONES: range(10, 20) if da == 'no_da' else [20, 30] if da == 'light_da' else [40, 70, 90],
			    P.KEY_MOMENTUM: 0.9,
			    P.KEY_L2_PENALTY: l2_penalties[ds + da_names[da]],
				P.KEY_DROPOUT_P: 0.5,
			}
			
			