from neurolab import params as P
from neurolab import utils
from neurolab.model import Model
import hebb as H
from hebb import functional as HF


class Net(Model):
	# Layer names
	FC = 'fc'
	CLASS_SCORES = 'class_scores' # Name of the classification output providing the class scores
	
	def __init__(self, config, input_shape=None):
		super(Net, self).__init__(config, input_shape)
		
		self.NUM_CLASSES = P.GLB_PARAMS[P.KEY_DATASET_METADATA][P.KEY_DS_NUM_CLASSES]
		self.ALPHA = config.CONFIG_OPTIONS.get(P.KEY_ALPHA, 1.)
		
		# Here we define the layers of our network
		
		self.fc = H.HebbianConv2d(
			in_channels=self.get_input_shape()[0],
			out_channels=self.NUM_CLASSES,
			kernel_size=(self.get_input_shape()[1], self.get_input_shape()[2]) if len(self.get_input_shape()) >= 3 else 1,
			lrn_sim=HF.raised_cos2d_pow(2),
			lrn_act=HF.identity,
			lrn_cmp=H.Competitive(),
			out_sim=HF.vector_proj2d,
			out_act=HF.identity,
			out_cmp=None,
			gating=H.HebbianConv2d.GATE_BASE,
			upd_rule=H.HebbianConv2d.UPD_RECONSTR,
			reconstruction=H.HebbianConv2d.REC_QNT_SGN,
			reduction=H.HebbianConv2d.RED_W_AVG,
			alpha=self.ALPHA,
		)  # input_shape-shaped x, NUM_CLASSES-dimensional output (one per class)
	
	# Here we define the flow of information through the network
	def forward(self, x):
		out = {}
		
		# Linear FC layer, outputs are the class scores
		fc_out = self.fc(x if len(self.get_input_shape()) >= 3 else x.view(x.size(0), x.size(1), 1, 1)).view(-1, self.NUM_CLASSES)
		
		# Build dictionary containing outputs from convolutional and FC layers
		out[self.FC] = fc_out
		out[self.CLASS_SCORES] = {P.KEY_CLASS_SCORES: fc_out}
		return out
	
	def set_teacher_signal(self, y):
		if isinstance(y, dict): y = y[P.KEY_LABEL_TARGETS]
		if y is not None: y = utils.dense2onehot(y, self.NUM_CLASSES)
		self.fc.set_teacher_signal(y)
		
	def local_updates(self):
		self.fc.local_update()

