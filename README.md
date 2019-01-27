# Hyper Parameters Optimization using Surrogate_Models
##A simple implementation of an Hyper Parameters optimization algorithm using surrogate models.


In this repo is implemented a simple algorithm for NN hyper parameter optimization.
Starting from few points already tested, the model build a surrogate based on RBF.
New hyper parameters are generated minimizing the "bumpiness" of the actual surrogate.
For solving the bumpiness problem, this implementation uses a Memetic Differential Evolution as global optimization algorithm which is equipped with L-BFGS-B for the local search phase.
