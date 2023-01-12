lrn_rules = ['hpca', 'hpca_dts', 'hpcat', 'hpcat_dts', 'hpcat_ada', 'hpcat_ada_dts',
             'ica', 'ica_dts', 'ica_nrm', 'ica_nrm_dts',
             '1wta', '1wta_dts', '5wta', '5wta_dts', 'eswta', 'eswta_dts', 'pswta', 'pswta_dts']
lrn_rule_keys = {'hpca': 'hpca', 'hpca_dts': 'hpca', 'hpcat': 'hpcat', 'hpcat_dts': 'hpcat', 'hpcat_ada': 'hpcat_ada', 'hpcat_ada_dts': 'hpcat_ada',
                 'ica': 'ica', 'ica_dts': 'ica', 'ica_nrm': 'ica_nrm', 'ica_nrm_dts': 'ica_nrm',
                 '1wta': 'hwta', '1wta_dts': 'hwta', '5wta': 'hwta', '5wta_dts': 'hwta', 'eswta': 'hwta', 'eswta_dts': 'hwta', 'pswta': 'hwta', 'pswta_dts': 'hwta'}
lrn_rule_dts = {lrn_rule: lrn_rule.endswith('_dts') for lrn_rule in lrn_rules}
lrn_rule_competitive_act = {'hpca': None, 'hpca_dts': None, 'hpcat': None, 'hpcat_dts': None, 'hpcat_ada': None, 'hpcat_ada_dts': None,
                            'ica': None, 'ica_dts': None, 'ica_nrm': None, 'ica_nrm_dts': None,
                            '1wta': 'hebb.functional.kwta', '1wta_dts': 'hebb.functional.kwta',
                            '5wta': 'hebb.functional.kwta', '5wta_dts': 'hebb.functional.kwta',
                            'eswta': 'hebb.functional.esoftwta', 'eswta_dts': 'hebb.functional.esoftwta',
                            'pswta': 'hebb.functional.psoftwta', 'pswta_dts': 'hebb.functional.psoftwta'}
lrn_rule_k = {'hpca': 0, 'hpca_dts': 0, 'hpcat': 0, 'hpcat_dts': 0, 'hpcat_ada': 0, 'hpcat_ada_dts': 0,
              'ica': 0, 'ica_dts': 0, 'ica_nrm': 0, 'ica_nrm_dts': 0,
              '1wta': 1, '1wta_dts': 1, '5wta': 5, '5wta_dts': 5, 'eswta': .02, 'eswta_dts': .02, 'pswta': .05, 'pswta_dts': .05}
hebbreg_coeffs = [-1e-3, -5e-4, -1e-4, -5e-5, 0., 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

datasets = ['mnist', 'cifar10', 'cifar100', 'tinyimagenet', 'imagenet']
data_managers = {'mnist': 'neurolab.data.MNISTDataManager', 'cifar10': 'neurolab.data.CIFAR10DataManager', 'cifar100': 'neurolab.data.CIFAR100DataManager',
                 'tinyimagenet': 'neurolab.data.TinyImageNetDataManager', 'imagenet': 'neurolab.data.ImageNetDataManager'}
da_strategies = ['no_da', 'light_da', 'hard_da']
da_names = {'no_da': '', 'light_da': '_lda', 'hard_da': '_hda'}
da_managers = {'no_da': None, 'light_da': 'neurolab.data.LightCustomAugmentManager', 'hard_da': 'neurolab.data.CustomAugmentManager'}
da_mult = {'no_da': 1, 'light_da': 2, 'hard_da': 5}
tot_trn_samples = {'mnist': 50000, 'cifar10': 40000, 'cifar100': 40000, 'tinyimagenet': 90000, 'imagenet': 1200000}
input_shapes = {'mnist': (3, 32, 32), 'cifar10': (3, 32, 32), 'cifar100': (3, 32, 32), 'tinyimagenet': (3, 32, 32), 'imagenet': (3, 210, 210)}
batch_sizes = {'mnist': 64, 'cifar10': 64, 'cifar100': 64, 'tinyimagenet': 64, 'imagenet': 32}
num_layers = {'mnist': 6, 'cifar10': 6, 'cifar100': 6, 'tinyimagenet': 6, 'imagenet': 10}
net_outputs = {'mnist': 'fc6', 'cifar10': 'fc6', 'cifar100': 'fc6', 'tinyimagenet': 'fc6', 'imagenet': 'fc10'}
hebb_lrn_rates = {'hpca': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-4},
                  'hpcat': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-4},
                  'hpcat_ada': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-4},
                  'ica': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-4},
                  'ica_nrm': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-4},
                  'hwta': {'mnist': 1e-3, 'cifar10': 1e-3, 'cifar100': 1e-3, 'tinyimagenet': 1e-3, 'imagenet': 1e-3}}
l2_penalties = {'mnist': 5e-2, 'mnist_lda': 2.5e-2, 'mnist_hda': 2.5e-2,
                'cifar10': 5e-2, 'cifar10_lda': 2.5e-2, 'cifar10_hda': 2.5e-2,
                'cifar100': 1e-2, 'cifar100_lda': 5e-3, 'cifar100_hda': 5e-3,
                'tinyimagenet': 5e-3, 'tinyimagenet_lda': 2.5e-3, 'tinyimagenet_hda': 2.5e-3,
                'imagenet': 1e-3, 'imagenet_lda': 5e-4, 'imagenet_hda': 5e-4}
samples_per_class = {'mnist': 5000, 'cifar10': 4000, 'cifar100': 400, 'tinyimagenet': 450, 'imagenet': 1200}
retr_num_samples = {'mnist': tot_trn_samples['mnist'], 'cifar10': tot_trn_samples['cifar10'], 'cifar100': tot_trn_samples['cifar100'], 'tinyimagenet': tot_trn_samples['tinyimagenet'], 'imagenet': tot_trn_samples['imagenet']}
retr_num_rel = {'mnist': samples_per_class['mnist'], 'cifar10': samples_per_class['cifar10'], 'cifar100': samples_per_class['cifar100'], 'tinyimagenet': samples_per_class['tinyimagenet'], 'imagenet': samples_per_class['imagenet']}
retr_k = {ds: [100, retr_num_rel[ds]] for ds in datasets}
retr_num_nyst = {ds: 0 for ds in datasets}
smpleff_regimes = {
	'mnist': [500, 1000, 1500, 2000, 2500, 5000, 12500],
	'cifar10': [400, 800, 1200, 1600, 2000, 4000, 10000],
	'cifar100': [400, 800, 1200, 1600, 2000, 4000, 10000],
	'tinyimagenet': [900, 1800, 2700, 3600, 4500, 9000, 22500],
	'imagenet': [12000, 24000, 36000, 48000, 60000, 120000, 300000],
}