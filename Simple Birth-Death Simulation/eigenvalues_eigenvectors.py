# -*- coding: utf-8 -*-

import numpy as np

b = 0.01       # per-capita birth rate
c = 0.005      # per-capita death rate

A = np.array(([-b, c, 0],
               [b, -(b+c), c],
               [0, b, -c]))

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("Eigenvectors:", eigenvectors)

def general_solution(t, c1, c2, c3, eigenvalues, eigenvectors):
    lambda1, lambda2, lambda3 = eigenvalues
    v1=eigenvectors[:, 0]
    v2=eigenvectors[:, 1]
    v3=eigenvectors[:, 2]
    x_t=c1*np.exp(lambda1*t)*v1+c2*np.exp(lambda2*t)*v2+c3*np.exp(lambda3*t)*v3
    return x_t

"""


Spyder Editor


"""

