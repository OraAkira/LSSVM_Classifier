# *-* coding: utf-8 *-*

import math
import numpy as matrix

def Classifier(x, y, length, sigma, gamma):
    """ LS-SVM classifier.

    Args:
        x: A matrix which rows means a data item
        y: A matrix which rows means a category
        length: A integer means the numbers of train data
        sigma: A integer means kernel function's parameter
        gamma: A integer means relaxation
    
    Returns:
        A matrix mean the answer of classifier
    """
    para = Train(x[0 : length], y[0 : length], sigma, gamma)
    beta = para[0]
    alpha = para[1 : length + 1]
    answer = matrix.zeros(len(x))
    for i in range(len(x)):
        ans[i] = beta
        for j in range(length):
            ans[i] += alpha[j] * y[j] * Kernel(x[i], x[j], sigma)
    return ans

def Kernel(x, y, sigma, gamma):
    delta = x - y
    sumSqure = float(delta.dot(delta.T))
    result = math.exp( -0.5 * sumSqure / (sigma ** 2))
    return result

def Train(x, y, sigma, gamma):
    length = len(x) + 1

    A = matrix.zeros(length, length)
    A[0][0] = 0
    A[0, 1:length] = y.T
    A[1:length, 0] = y
    A[1:length, 1:length] = Omega(x, y, sigma) + matrix.eye(length - 1) / gamma

    B = matrix.ones(length, 1)
    B[0][0] = 0

    return matrix.linalg.solve(A, B)

def Omega(x, y, sigma):
    length = len(x)
    omega = matrix.zeros(length, length)
    for i in range(length):
        for j in range(length):
            omega[i, j] = y[i] * y[j] * Kernel(x[i], x[j], sigma)
    return omega
