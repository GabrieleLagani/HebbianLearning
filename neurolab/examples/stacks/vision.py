import neurolab.params as P


stack_fc_on_layers = [{P.KEY_STACK_CONFIG: 'neurolab.examples.configs.vision.config_6l', P.KEY_STACK_SEEDS: [0], P.KEY_STACK_DATASEEDS: [200]}]
stack_fc_on_layers += [{P.KEY_STACK_CONFIG: 'neurolab.examples.configs.vision.fc_on_layer[' + str(l) + ']', P.KEY_STACK_SEEDS: [300], P.KEY_STACK_TOKENS: ['0'], P.KEY_STACK_DATASEEDS: [200]} for l in range(1, 6)]

