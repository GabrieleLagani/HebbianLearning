Pytorch implementation of Hebbian learning algorithms to train
deep convolutional neural networks.
A neural network model is trained on various datasets both using 
Hebbian algorithms and SGD in order to compare the results.
Hybrid models with some layers trained by Hebbian learning and other 
layers trained by SGD are studied.
A semi-supervised approach is also considered, where unsupervised
Hebbian learning is used to pre-train internal layers of a DNN, 
followed by SGD fine-tuning of a final linear classifier. The approach
performs better than simple end-to-end backprop training in low
sample efficiency scenarios.

We also introduce a preliminary version of the `neurolab` package, a
simple, extensible, Pytorch-based framework for deep learning 
experiments, providing functionalities for handling experimental 
configurations, reproducibility, checkpointing/resuming experiment state, 
hyperparameter search, and other utilities.

This work is a continuation of 
https://github.com/GabrieleLagani/HebbianLearningThesis 
and https://github.com/GabrieleLagani/HebbianPCA,
containing the latest updates and new Hebbian learning algorithms: 
Hebbian WTA variants (k-WTA, soft-WTA, ...), Hebbian PCA, Hebbian ICA.

In order to launch an experiment session, type:  
`PYTHONPATH=<project root> python <project root>/runexp.py --config <dotter.path.to.config.object> --mode <mode> --device <device> --clearhist --restart`  
Where `<dotter.path.to.config.object>` is the path, in dotted notation,
to a dictionary (which can be defined anywhere in the code) containing
the configuration parameters that you want to use for your experiment;
`<mode>` is either `train` or `test`; `<device>` is, for example, `cpu`
or `cuda:0`; the `--clearhist` flag can be used if you don't need
to keep old checkpoint files saved on disk; and the `--restart` flag 
can be used to restart an experiment from scratch, instead of resuming 
from a checkpoint.

You can also use:
`PYTHONPATH=<project root> python <project root>/runstack.py --stack <dotter.path.to.stack.object> --mode <mode> --device <device> --clearhist --restart`  
to run full stack of experiments (i.e. a list, which can be defined anywhere in the code).

Example:
`python runstack.py --stack stacks.vision.all[cifar10] --mode train --device cuda:0 --clearhist`


Author: Gabriele Lagani - gabriele.lagani@gmail.com