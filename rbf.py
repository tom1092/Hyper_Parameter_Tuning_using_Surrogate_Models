import numpy as np


class Problem:

    def __init__(self, dimension, bounds):
        self.dimension = dimension
        self.bounds = bounds

    def __call__(self, x):
        raise NotImplementedError('Subclasses have to override this method')


class RBF(Problem):

    def __init__(self, dimension, bounds, nodes, f, gamma):
        Problem.__init__(self, dimension, bounds)
        self.nodes = nodes
        self.f = f
        self.gamma = gamma
        self.phi, self.l = self.solve_rbf()

    # Build and solve the interpolation problem returning coefficients matrix and lambdas
    def solve_rbf(self):
        phi = np.ndarray(shape=(len(self.nodes), len(self.nodes)), dtype='float32')
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                phi[i, j] = np.exp(-self.gamma*np.linalg.norm(self.nodes[i]-self.nodes[j])**2)

        return phi, np.linalg.solve(phi, self.f)

    # Return the value of the fitting in x
    def __call__(self, x):
        radial = np.array([np.exp(-self.gamma*np.linalg.norm(x-self.nodes[j])**2) for j in range(len(self.nodes))]).astype('float32')

        return np.dot(radial, self.l)


class BumpinessProblem(Problem):

    def __init__(self, dimension, bounds, f_guess, rbf):
        Problem.__init__(self, dimension, bounds)
        self.f_guess = f_guess
        self.rbf = rbf

    # Build the bumpiness function and return its value in x_hat
    def __call__(self, x_hat):
        g_hat = np.linalg.det(self.rbf.phi)
        contributes = np.array(
            [np.exp(-self.rbf.gamma * np.linalg.norm(x_hat - self.rbf.nodes[j]) ** 2) for j in range(len(self.rbf.nodes))]).astype(
            'float32')
        M = np.ndarray(shape=(len(self.rbf.nodes) + 1, len(self.rbf.nodes) + 1), dtype='float32')
        M[0:len(self.rbf.nodes), 0:len(self.rbf.nodes)] = self.rbf.phi
        M[-1, 0:-1] = contributes.T
        M[0:-1, -1] = contributes
        M[-1, -1] = 1

        # Avoid numerical problems (If M is bad conditioned its determinant can be a negative, very small number)
        if np.linalg.det(M) > 0:
            g_hat /= np.linalg.det(M)
            bumpiness = ((self.f_guess - self.rbf(x_hat)) ** 2) * g_hat
        else:
            bumpiness = np.inf

        return bumpiness










