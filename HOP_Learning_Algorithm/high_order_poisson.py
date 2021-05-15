#AlgName.py
#
#This script shows how to code a new graph-based learning
#algorithm and incorporate it into ssl_trials to 
#compare to other SSL algorithms. 

import graphlearning as gl
import numpy as np
import scipy.sparse as sparse
import os

#Below we define a new ssl algorithm.  The name must be 'ssl'. The file name 'alg_name.py' can be anything you like,
#provided it is **lower case**. The basename 'alg_name' without the extension '.py' is the algorithm name for 
#gl.ssl_trials and gl.graph_ssl.
#
#The algorithm below is Higher-Order Poisson Learning with a soft label fidelity
#min_u u^T L u + lam*(u-g)^2
#min_u u^t L^m u + lam * poisson source term
def ssl(W,I,g,params):
#The inputs/outputs must match exactly:
#Inputs:
#  W = sparse weight matrix 
#  I = indices of labeled nodes in graph
#  g = integer labels corresponding to labels
#  params = a Python dictionary with any custom parameters for the algorithm
#Output:
#  u = kxn array of probability or one-hot vectors for each node
    
    #Size of graph
    n = W.shape[0]
    
    ##Get the Poisson Source Term
    #Labels to vector and correct position
    J = np.zeros(n,)
    K = np.ones(n,)*g[0]
    J[I] = 1
    K[I] = g
    Kg,_ = gl.LabelsToVec(K)
    Kg = Kg*J
    
    #Poisson source term
    c = np.sum(Kg,axis=1)/len(I)
    b = np.transpose(Kg)
    b[I,:] = b[I,:]-c

    #Regularization parameter
    N = params['N']
    vals = params['vals'][:N]
    vecs = params['vecs'][:,:N]
    m = params['power']

    #Spectral Approximation
    inverse_vals = (1/vals[1:]) ** m
    #inverse_vals_full = np.concatenate((np.array([[0]]), inverse_vals), axis = None)
    vecs = vecs[:, 1:]
    length = inverse_vals.shape[0]
    D = sparse.spdiags(inverse_vals, 0, length, length)
    x_N = (vecs @ D) @ (vecs.T @ b)
    
    return x_N.T

