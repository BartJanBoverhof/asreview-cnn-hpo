# ASReview CNN with hyperparameter optimisation 
This repository is a plugin containing a convolutional neural network implementation that may be utilised when doing a systematic review with [ASReview](https://github.com/asreview). This plugin includes two models, being (1) a base Convolutional Neural Network (CNN), and (2) a Naive Bayes (NB) - CNN combination model, starting with NB for the first X amount of iterations, and switching to a CNN hereafter. Both models make use of hyperparamater optimisation, which is repeated for every 300 iterations.  

## Getting started
Install the new classifier with:

```bash
pip install .
```

or

```bash
python -m pip install git+https://github.com/BartJanBoverhof/asreview-cnn-hpo.git
```

## Usage
The ``cnn base model`` is defined in [`asreviewcontrib/models/classifiers/cnn_base.py`](asreviewcontrib/models/classifiers/cnn_base.py) and can be used with `--model cnn-base`.

The ``nb-cnn switch model`` is defined in [`asreviewcontrib/models/classifiers/cnn_switch.py`](asreviewcontrib/models/classifiers/cnn_switch.py) and can be used with `--model cnn-switch`.

## License 
Apache-2.0 License 