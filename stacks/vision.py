import neurolab.params as P
from configs.vision.meta import *


lrn_rules = ['1wta', '5wta', 'eswta', 'pswta', 'hpca', 'ica']
layers = {'mnist': range(1, 6), 'cifar10': range(1, 6), 'cifar100': range(1, 6), 'tinyimagenet': range(1, 6), 'imagenet': [5, 7, 9]}

SEEDS1 = range(0, 5)
SEEDS2 = range(300, 305)
SEEDS3 = range(400, 405)
SEEDS4 = range(500, 505)
DATASEEDS = range(200, 205)

gdes = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
hebb = {lrn_rule + '_' + ds + da_names[da]: [] for lrn_rule in lrn_rules for ds in datasets for da in da_strategies}
vae = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
hybrid = {lrn_rule + '_' + ds + da_names[da]: [] for lrn_rule in lrn_rules for ds in datasets for da in da_strategies}
smpleff_gdes = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
smpleff_hebb = {lrn_rule + '_' + ds + da_names[da]: [] for lrn_rule in lrn_rules for ds in datasets for da in da_strategies}
smpleff_vae = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
sk_gdes = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
sk_hebb = {lrn_rule + '_' + ds + da_names[da]: [] for lrn_rule in lrn_rules for ds in datasets for da in da_strategies}
sk_vae = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
smpleff_sk_gdes = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
smpleff_sk_hebb = {lrn_rule + '_' + ds + da_names[da]: [] for lrn_rule in lrn_rules for ds in datasets for da in da_strategies}
smpleff_sk_vae = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
all = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}

hebbreg = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
hebbreg_sk = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
vaereg = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}
vaereg_sk = {ds + da_names[da]: [] for ds in datasets for da in da_strategies}


for ds in datasets:
	for da in da_strategies:
		gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.config_base_gdes[' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
		gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc2_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.gdes_fc2_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		for lrn_rule in lrn_rules:
			gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc2_on_gdes_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(num_layers[ds] - 2, num_layers[ds] - 1)] #for l in layers[ds]]
			#gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.hebb_fc2_on_gdes_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.prec_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.prec_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.knn_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.knn_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.svm_on_gdes_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.gdes.svm_on_gdes_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]

		for lrn_rule in lrn_rules:
			hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.config_base_hebb[' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
			hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			#hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			#hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(num_layers[ds] - 2, num_layers[ds] - 1)] #for l in layers[ds]]
			#hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.gdes_fc2_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			#hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			#hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.hebb_fc2_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.prec_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.prec_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.knn_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.knn_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.svm_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebb.svm_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
			
		vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.config_base_vae[' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS}]
		vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc2_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.gdes_fc2_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#for lrn_rule in lrn_rules:
		#   vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc2_on_vae_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		#   vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.hebb_fc2_on_vae_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.prec_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.prec_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.knn_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.knn_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.svm_on_vae_layer[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.vae.svm_on_vae_layer_ft[' + str(l) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in layers[ds]]
		
		for lrn_rule in lrn_rules:
			hybrid[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.gdes_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds] - 2)]
			hybrid[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.hebb_on_gdes_layer[' + str(l) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for l in range(1, num_layers[ds] - 2)]
			for l1 in range(1, num_layers[ds] - 1):
				for l2 in range(l1 + 1, num_layers[ds]):
					hybrid[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hybrid.ghg[' + str(l1) + '_' + str(l2) + '_' + lrn_rule + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS1, SEEDS2)], P.KEY_STACK_DATASEEDS: DATASEEDS}]
			
		smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.config_base_gdes[' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for l in layers[ds]:
			smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#for lrn_rule in lrn_rules:
			#	smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_gdes_layer_pt1[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#	smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_gdes_layer_pt2[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS1, SEEDS2)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#	smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_gdes_layer_ft_pt1[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#	smpleff_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_gdes_layer_ft_pt2[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS4, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS2, SEEDS3)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_gdes_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_sk_gdes[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_gdes_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for lrn_rule in lrn_rules:
			for l in layers[ds]:
				smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS1, SEEDS2)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_hebb_layer_ft_pt1[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				#smpleff_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_hebb_layer_ft_pt2[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS4, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS2, SEEDS3)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_hebb_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
				smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_hebb_layer_ft[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
		for l in layers[ds]:
			smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.gdes_fc2_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#for lrn_rule in lrn_rules:
			#	smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_vae_layer[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS1, SEEDS2)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#	smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_vae_layer_ft_pt1[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			#	smpleff_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.hebb_fc2_on_vae_layer_ft_pt2[' + str(l) + '_' + lrn_rule + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS4, P.KEY_STACK_TOKENS: ['{},{}'.format(i, j) for i, j in zip(SEEDS2, SEEDS3)], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.prec_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.knn_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_vae_layer[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
			smpleff_sk_vae[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.smpleff.svm_on_vae_layer_ft[' + str(l) + '_' + ds + '_' + str(n) + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS3, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS2], P.KEY_STACK_DATASEEDS: DATASEEDS} for n in smpleff_regimes[ds]]
	
	
		all[ds + da_names[da]] += gdes[ds + da_names[da]]
		#all[ds + da_names[da]] += vae[ds + da_names[da]]
		for lrn_rule in lrn_rules: all[ds + da_names[da]] += hebb[lrn_rule + '_' + ds + da_names[da]]
		for lrn_rule in lrn_rules: all[ds + da_names[da]] += hybrid[lrn_rule + '_' + ds + da_names[da]]
		#all[ds + da_names[da]] += smpleff_gdes[ds + da_names[da]]
		#all[ds + da_names[da]] += smpleff_vae[ds + da_names[da]]
		for lrn_rule in lrn_rules: all[ds + da_names[da]] += smpleff_hebb[lrn_rule + '_' + ds + da_names[da]]
		#all[ds + da_names[da]] += sk_gdes[ds + da_names[da]]
		#all[ds + da_names[da]] += sk_vae[ds + da_names[da]]
		#for lrn_rule in lrn_rules: all[ds + da_names[da]] += sk_hebb[lrn_rule + '_' + ds + da_names[da]]
		#all[ds + da_names[da]] += smpleff_sk_gdes[ds + da_names[da]]
		#all[ds + da_names[da]] += smpleff_sk_vae[ds + da_names[da]]
		#for lrn_rule in lrn_rules: all[ds + da_names[da]] += smpleff_sk_hebb[lrn_rule + '_' + ds + da_names[da]]
	
		for lrn_rule in lrn_rules: hebbreg[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebbreg.config_hebbreg[' + lrn_rule + '_a' + str(a) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS} for a in hebbreg_coeffs if a != 0. or lrn_rule == lrn_rules[0]]
		for lrn_rule in lrn_rules: hebbreg_sk[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebbreg.prec_on_hebbreg[' + lrn_rule + '_a' + str(a) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for a in hebbreg_coeffs if a != 0. or lrn_rule == lrn_rules[0]]
		vaereg[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebbreg.config_vaereg[a' + str(a) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS1, P.KEY_STACK_DATASEEDS: DATASEEDS} for a in hebbreg_coeffs if a != 0. or lrn_rule == lrn_rules[0]]
		vaereg_sk[ds + da_names[da]] += [{P.KEY_STACK_CONFIG: 'configs.vision.hebbreg.prec_on_vaereg[a' + str(a) + '_' + ds + da_names[da] + ']', P.KEY_STACK_SEEDS: SEEDS2, P.KEY_STACK_TOKENS: [str(i) for i in SEEDS1], P.KEY_STACK_DATASEEDS: DATASEEDS} for a in hebbreg_coeffs if a != 0. or lrn_rule == lrn_rules[0]]

