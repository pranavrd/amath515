import numpy as np
from solvers import BFGS

def get_rosenbrock(d):
    def f(x):
        coupled_term = 100*(x[1:]**2-x[:-1])**2
        diagonal_term = (x  - 1.)**2
        return np.sum(coupled_term) + np.sum(diagonal_term)
    
    def gradf(x):
        grad = 2.0 * (x - 1.0)
        diffs = x[1:]**2 - x[:-1]
        grad[:-1] += -200.0 * diffs
        grad[1:]  += 400.0 * x[1:] * diffs
        return grad
    
    def hessf(x):
        diag_main = np.full(d, 2.0)
        diag_main[:-1] += 200.0
        diag_main[1:] += 1200.0 * x[1:]**2 - 400.0 * x[:-1]
        off_diag = np.zeros(d - 1)
        off_diag = -400.0 * x[1:]
        H = (
            np.diag(diag_main) +
            np.diag(off_diag, k=1) +
            np.diag(off_diag, k=-1)
        )
        return H
    x0 = -1 * np.ones(d)
    
    return f,gradf,hessf,x0

# New function for testing optimization techniques - zakharov : it is a plate shaped function with a global minimum but no local minimum
def get_zakharov(d):
    def f(x):
        term1 = np.sum(x**2)
        term2 = np.sum(0.5 * np.arange(1, d+1) * x)
        return term1 + term2**2 + term2**4

    def gradf(x):
        term1 = 2 * x
        term2 = np.sum(0.5 * np.arange(1, d+1) * x)
        term3 = 0.5 * np.arange(1, d+1)
        return term1 + 2 * term2 * term3 + 4 * term2**3 * term3

    def hessf(x):
        term2 = np.sum(0.5 * np.arange(1, d+1) * x)
        term3 = 0.5 * np.arange(1, d+1)
        H = np.diag(2 * np.ones(d)) + 2 * np.outer(term3, term3) + 12 * term2**2 * np.outer(term3, term3)
        return H

    x0 = 0.5 * np.ones(d)
    
    return f, gradf, hessf, x0

#MNIST Logistic regression

def get_lgt_obj(lam_lgt):
    mnist_data = np.load('mnist01.npy', allow_pickle=True)
    #
    A_lgt = mnist_data[0]
    b_lgt = mnist_data[1]

    # define function, gradient and Hessian
    def lgt_func(x):
        y = A_lgt.dot(x)
        return np.sum(np.log(1.0 + np.exp(y))) - b_lgt.dot(y) + 0.5 * lam_lgt * np.sum(x**2)
    #
    def lgt_grad(x):
        y = A_lgt.dot(x)
        z = 1.0 / (1.0 + np.exp(-y)) - b_lgt
        return A_lgt.T.dot(z) + lam_lgt * x
    #
    def lgt_hess(x):
        y = A_lgt.dot(x)
        z = np.exp(-y) / (1.0 + np.exp(-y))**2
        return A_lgt.T.dot(np.diag(z).dot(A_lgt)) + lam_lgt * np.eye(x.size)
    
    x0 = np.zeros(A_lgt.shape[1])
    return lgt_func, lgt_grad, lgt_hess, x0


