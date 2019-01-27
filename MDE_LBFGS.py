# This file implements the Memetic Differential Evolution method with
# local search using L-BFGS

from scipy.optimize import minimize
import random
import numpy as np


class MDE_LBFGS:

    def __init__(self, dimension, bounds, pop_size=100, gen=100, f=0.1, cr=0.9):

        self.gen = gen
        self.f = f
        self.cr = cr
        self.dimension = dimension
        self.pop_size = pop_size
        self.bounds = bounds

    # Just make a projection on the feasible set with box constraints
    def project_on_feasible_set(self, x):

        for i in range(self.dimension):
            x[x[i] < self.bounds[i][0]] = self.bounds[i][0]
            x[x[i] > self.bounds[i][1]] = self.bounds[i][1]

    # Define the MDE algorithm starting with a uniform population sampled from the constraints box.
    # Use L-BFGS-B for the local search
    def solve(self, problem):

        pop = np.ndarray(shape=(self.pop_size, self.dimension), dtype='float32')
        for i in range(self.dimension):
            pop[:, i] = np.random.uniform(self.bounds[i][0], self.bounds[i][1], self.pop_size)
        best = pop[0]
        print ("Minimizing surrogate with MDE + L-BFGS-B...")
        for generation in range(self.gen):

            for i in range(self.pop_size):
                i_star = random.randint(0, self.dimension - 1)
                indexes = list(range(0, self.pop_size))
                indexes.remove(i)
                k_indexes = random.sample(indexes, 3)
                trial = np.array([pop[i, j] if j != i_star and np.random.random_sample() < self.cr else pop[k_indexes[0], j] + self.f * (pop[k_indexes[1], j]-pop[k_indexes[2], j]) for j in range(self.dimension)])
                self.project_on_feasible_set(trial)
                trial = minimize(problem, trial, method='L-BFGS-B', bounds=self.bounds).x
                if problem(trial) < problem(pop[i]):
                    pop[i] = trial
                if problem(pop[i]) < problem(best):
                    best = pop[i]
        return best

