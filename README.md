# ASReview CNN with hyperparameter optimisation 
This repository is a plugin containing a convolutional neural network (CNN) implementation that may be utilised when doing a systematic review with [ASReview](https://github.com/asreview). This plugin includes a model combining a Naive Bayes (NB) and CNN, starting with NB for the first X amount of iterations, and switching to a CNN hereafter. The current switchpoint is set at 500 iterations, but may be changed by respecified user. The CNN makes use of hyperparamater optimisation, which is repeated for every 300 iterations.  

The preferred feature extraction strategy for this model is [wide-doc2vec](https://github.com/JTeijema/) asreview-plugin-wide-doc2vec/)

## Getting started
Install the new classifiers with:

```bash
pip install .
```

or

```bash
python -m pip install git+https://github.com/BartJanBoverhof/asreview-cnn-hpo.git
```

## Usage
The ``nb-cnn switch model`` is defined in [`asreviewcontrib/models/classifiers/cnn_switch.py`](asreviewcontrib/models/classifiers/cnn_switch.py) and can be used with `--model cnn-switch`.

## Performance 
A simulation study assessubg the performance of this model can be found [here](link.com). In short, no direct evidence was found in favor of the current implementation of the `cnn-switch` with `wide-doc2vec` to outperform already implemented models such `nb` and `lr`, however, a differently optimised model may provide to show potential (see also: discussion section of the aformentioned report).

## License 
Apache-2.0 License 