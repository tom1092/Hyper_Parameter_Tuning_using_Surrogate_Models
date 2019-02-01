# Hyper Parameters Optimization using Surrogate Models
#### A simple implementation of an Hyper Parameters optimization algorithm using surrogate models.


In this repo is implemented a simple algorithm for NN hyper parameter optimization.
Starting from few points already tested, the model builds a surrogate based on RBF.

New hyper parameters are generated minimizing the "bumpiness" of the actual surrogate.

For solving the bumpiness problem, this implementation uses a Memetic Differential Evolution as global optimization algorithm which is equipped with L-BFGS-B for the local search phase.


### Dependencies

Package name | Version
------------ | -------------
[Python](https://www.python.org/) | 2.7 
[Numpy](http://www.numpy.org/) | 1.13.3 or higher
[Scipy](https://www.scipy.org/) | 0.19.1 or higher
[Pandas](https://pandas.pydata.org/) | 0.23.4 or higher
[Tensorflow](https://www.tensorflow.org/) | 1.7.0 or higher
[Keras](https://keras.io/) | 2.2.4 or higher



In [demo.py](https://github.com/tom1092/Hyper_Parameters_Optimization_using_Surrogate_Models/blob/master/demo.py) is implemented a simple NN that you can train on a binary classification problem respect to the [sonar dataset](https://www.kaggle.com/adx891/sonar-data-set). There are already 3 combinations of hyper parameter tested from which the algorithm starts to build the surrogate model. 
