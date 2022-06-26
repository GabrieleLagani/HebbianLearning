HEBB_UPD_GRP = 128 # In the Hebbian module, how many kernels to update in parallel on GPU. Larger is faster, but make it smaller to avoid memory overflow.
HEBB_FASTHEBB = True # Whether to use fast hebbian update computation based on matmul
HEBB_REORDMULT = True # Whether to use multiplication reordering with early batch-wise aggregation
KEY_VAE_NUM_LATENT_VARS = 'vae_num_latent_vars'
KEY_COMPETITIVE_ACT = 'competitive_act'
KEY_COMPETITIVE_K = 'competitive_k'
KEY_ACT_COMPLEMENT_INIT = 'act_complement_init'
KEY_ACT_COMPLEMENT_RATIO = 'act_complement_ratio'
KEY_ACT_COMPLEMENT_ADAPT = 'act_complement_adapt'
KEY_ACT_COMPLEMENT_GRP = 'act_complement_grp'
KEY_LRN_SIM = 'lrn_sim'
KEY_LRN_SIM_B = 'lrn_sim_b'
KEY_LRN_SIM_S = 'lrn_sim_s'
KEY_LRN_SIM_P = 'lrn_sim_p'
KEY_LRN_SIM_EXP = 'lrn_sim_exp'
KEY_LRN_SIM_NC = 'lrn_sim_nc'
KEY_LRN_ACT = 'lrn_act'
KEY_LRN_ACT_SCALE_IN = 'lrn_act_scale_in'
KEY_LRN_ACT_SCALE_OUT = 'lrn_act_scale_out'
KEY_LRN_ACT_OFFSET_IN = 'lrn_act_offset_in'
KEY_LRN_ACT_OFFSET_OUT = 'lrn_act_offset_out'
KEY_LRN_ACT_P = 'lrn_act_p'
KEY_OUT_SIM = 'out_sim'
KEY_OUT_SIM_B = 'out_sim_b'
KEY_OUT_SIM_S = 'out_sim_s'
KEY_OUT_SIM_P = 'out_sim_p'
KEY_OUT_SIM_EXP = 'out_sim_exp'
KEY_OUT_SIM_NC = 'out_sim_nc'
KEY_OUT_ACT = 'out_act'
KEY_OUT_ACT_SCALE_IN = 'out_act_scale_in'
KEY_OUT_ACT_SCALE_OUT = 'out_act_scale_out'
KEY_OUT_ACT_OFFSET_IN = 'out_act_offset_in'
KEY_OUT_ACT_OFFSET_OUT = 'out_act_offset_out'
KEY_OUT_ACT_P = 'out_act_p'

