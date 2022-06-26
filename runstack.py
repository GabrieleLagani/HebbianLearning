import os
from neurolab import params as P
import neurolab.runstack
from version import __version__

if __name__ == '__main__':
	P.PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
	P.DATASETS_FOLDER = os.path.join(P.PROJECT_ROOT, 'datasets')
	P.STATS_FOLDER =  os.path.join(P.DATASETS_FOLDER, 'stats')
	P.RESULT_FOLDER = os.path.join(P.PROJECT_ROOT, 'results')
	P.HPEXP_RESULT_FOLDER = os.path.join(P.PROJECT_ROOT, 'hpresults')
	P.DEFAULT_STACK = 'stacks.base.stack_base'
	P.ADDITIONAL_INFO['Hebbian Learning vers.'] = __version__
	neurolab.runstack.main()