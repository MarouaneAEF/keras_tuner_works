
The objective is to test a discriminative model based on non RNN-like models. The reference paper, "Deep learning for time series classification: a review," is available at https://arxiv.org/pdf/1809.04356.pdf.

The model being chosen to work with is available at:  https://github.com/hfawaz/dl-4-tsc.

I suggest a training by working out hyperparameter tuning using the Keras Tuner library. It provides a comparison of different tuners applied to TimeSeriesClassification-like datasets. The available hyper-parameter fine-tuning strategies are:

    Random Search
    Hyperband
    Bayesian Search




 






