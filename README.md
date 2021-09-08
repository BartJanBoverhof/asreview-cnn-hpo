# ASReview CNN with hyperparameter optimisation 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5482149.svg)](https://doi.org/10.5281/zenodo.5482149)

This repository contains an extention for [ASReview](https://github.com/asreview) ![logo](https://raw.githubusercontent.com/asreview/asreview-artwork/e2e6e5ea58a22077b116b9c3d2a15bc3fea585c7/SVGicons/IconELAS/ELASeyes24px24px.svg "ASReview") containing a convolutional neural network (CNN) model that may be utilised during a systematic review with [ASReview](https://github.com/asreview). This extention includes a model combining Naive Bayes (NB) and CNN classifiers, starting with Naive Bayes for the first set amount of iterations, and switching to a CNN thereafter. The current switchpoint is set at 500 iterations, but can be adjusted by the user. This CNN makes use of hyperparamater optimisation, which is set to repeat every 300 iterations. The preferred feature extraction strategy for this model is the [wide-doc2vec](https://github.com/JTeijema/asreview-plugin-wide-doc2vec) feature extractor.

To read more about the rationale behind utilising two models within one systematic review, please consult the [simulation report](https://github.com/BartJanBoverhof/asreview-cnn-hpo/blob/main/report/asreview_report_bartjan.pdf).

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
A simulation study assessing the performance of this model can be found in the included [report](https://github.com/BartJanBoverhof/asreview-cnn-hpo/blob/main/report/asreview_report_bartjan.pdf). In short, no direct evidence was found in favor of the current implementation of the `cnn-switch` with `wide-doc2vec` to outperform already implemented models such `nb` and `lr`, however, a differently optimised model may provide to show potential (see also: discussion section of the report).

## License 
Apache-2.0 License 
