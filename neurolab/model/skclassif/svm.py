from sklearn.svm import LinearSVC

from .skclassif import SkClassif


class SVMClassifier(SkClassif):
	def __init__(self, config, input_shape=None):
		super(SVMClassifier, self).__init__(config, input_shape)
		
		self.clf = LinearSVC()

