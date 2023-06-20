import numpy as np
import core as cr
import matplotlib.pyplot as plt
import numpy.matlib

from scipy.linalg import toeplitz
from scipy.sparse import identity, diags


######################################################## TASK 2 ########################################################
########################################################################################################################


def getKernel1D(n, sig=0.03):
  '''
  GETKERNEL1D function to get discrete convolution operator
  for one-dimensional test problem
  
   input
     n       number of points
     sig     kernel width
  
   output
     K       kernel matrix
  '''

  h = 1/n[0] # compute spatial step size
  s = np.expand_dims(np.arange(h/2, 1-h/2+h, h), axis=0).T.conj() # compute coordinates

  #compute 1D kernel
  c = 1 / (np.sqrt(2*np.pi) * sig)
  ker = c * np.exp(-(s - h/2)**2 / (2 * sig**2))

  #compute kernel matrix
  K = h * toeplitz(ker)

  return K


def getDeconvSource1D(n):
  '''
   GETDECONVSOURCE1D get source (i.e., x_true)
   for deconvolution problem
  
   input:
     n     number of grid points
  
   output:
     x     data set
  '''

  # compute coordinate functions
  h = 1/n[0]
  s = np.expand_dims(np.arange(h/2, 1-h/2+h, h), axis=0).T.conj()

  # set up true solution x_true
  id1 = ((0.10<s) & (s<0.25))
  id2 = ((0.30<s) & (s<0.32))
  id3 = ((0.50<s) & (s<1.00))
  x = 0.75*id1 + 0.25*id2 + id3*np.sin(2*np.pi*s)**4
  x *= 50
  x = x / np.linalg.norm(x)

  return x, s

def scDeconvGCVLAP1D():
  #find optimal regularization parameter using GCV

  n = [80] # number of points
  delta = 2 # error level is 2%

  #get source for deconvolution problem
  x_true, s = getDeconvSource1D(n)
  K = getKernel1D(n) # get deconvolution operator

  #compute scaling for noise perturbation
  sig = delta * np.linalg.norm(K @ x_true) / (100*np.sqrt(n[0]))

  #compute additive noise
  eta = sig * np.random.normal(size=(n[0], 1))

  #compute observed data
  y_obs = K@x_true + eta

  #compute precision matrix of prior distribution
  L = cr.getLapMat(n, 1)

  #use generalized cross validation to compute optimal regularization
  #parameter linear invere problem
  alpha = cr.evalGCV(K, L, y_obs)
  print('optimal regularization parameter alpha =', alpha)
  
    
######################################################## TASK 3 ########################################################
#########################################################################################################################    
 
    
def scDeconvIGMRFEP1D():
  #one dimensional deconvolution based on independent increment
  #(anisotropic) IGMRF prior; the spatially dependent increment weights
  #are computed using an iterative algorithm; the optimal regularization
  #parameter is determined based on GCV

  n = [80] # number of points
  delta = 2 # error level is 2%
  gamma = 1e-3 # perturbation
  niter = 10 # number of iterations

  #get source for deconvolution problem
  x_true, s = getDeconvSource1D(n)
  K = getKernel1D(n) # get deconvolution operator

  #compute scaling for noise perturbation
  sig = delta * np.linalg.norm(K@x_true) / (100*np.sqrt(n[0]))
  np.random.seed(3)
  eta = sig * np.random.normal(size=(n[0],1))

  #compute observed data
  y_obs = K@x_true + eta

  #get one dimensional first order derivative operator
  D = cr.getGradMat(n, 1)

  #compute weight matrix for first iteration
  W = identity(n[0]-1)
  rhs = K.T.conj() @ y_obs

  for i in range(niter):
      #define prior precision matrix
      L = D.T @ W @ D # ADD YOUR CODE HERE

      #compute optimal regularization parameter
      alpha = cr.evalGCV(K, L, y_obs)
      print('iteration: {} (alpha = {})'.format(i, alpha))

      #compute tikhonov regularized solution
      x_alpha = np.linalg.inv(K.T@K + alpha*L) @ K.T @ y_obs # ADD YOUR CODE HERE

      #update weight matrix
      W = diags(np.squeeze(np.ones((n[0]-1, 1)) / np.sqrt(np.asarray(D@x_alpha)**2 + gamma*np.ones((n[0]-1, 1))) ) ) # ADD YOUR CODE HERE

  #plot reconstruction
  f, ax = plt.subplots(1, 2, figsize=(15,5))
  ax[0].plot(s, x_true, color='black', label='true solution')
  ax[0].scatter(s, y_obs, marker='x', color='red', label='observation')
  ax[0].legend()
  ax[1].plot(s, x_true, color='black', label='true solution')
  ax[1].plot(s, x_alpha, color='red', label='estimated source')
  ax[1].legend()

    
######################################################## TASK 6 #########################################################
##########################################################################################################################


def scDeconvGMRFLAP1D():
  #one dimensional deconvolution problem with independent increment
  #IGMRF prior: the associated Gaussian posterior density function is
  #sampled using a Cholesky factorization of the precision matrix
  # 
  #after the samples are computed, the sample mean is used as an estimator
  #of the unknown dataset; empirical quantiles are used to compute 95%
  #credibility intervals for every unknown

  ns = 1000 # number of samples to be drawn
  n = [80] # number of points
  delta = 2 # error level is 2%

  #get source for deconvolution problem
  x_true, s = getDeconvSource1D(n)
  K = getKernel1D(n) # get deconvolution operator

  #compute scaling for noise perturbation
  sig = delta * np.linalg.norm(K@x_true) / (100*np.sqrt(n[0]))

  #compute additive noise
  np.random.seed(3)
  eta = sig * np.random.normal(size=(n[0],1))

  #compute observed data
  y_obs = K @ x_true + eta

  #compute precision matrix of prior distribution
  L = cr.getLapMat(n, 1)

  #use generalized cross validation to compute optimal regularization
  #parameter linear invere problem
  alpha = cr.evalGCV(K, L, y_obs)
  print('optimal regularization parameter alpha =', alpha)

  #compute solution for "optimal" regularization parameter (parameter can
  #be found using various criteria; here we simply used trial and error)
  x_alpha = np.linalg.solve(((K.T.conj() @ K) + alpha * L), K.T.conj() @ y_obs)

  #estimate lambda and delta given solution of inverse problem
  r = y_obs - K @ x_alpha
  tau = 1 / np.var(r);
  beta = alpha * tau;

  #compute cholesky decomposition
  C = np.linalg.cholesky(tau * K.T @ K + beta * L)

  #compute MAP point by solving optimality condition
  x_map = tau * np.linalg.inv(C.T @ C) @ K.T @ y_obs

  #next draw samples from the posterior distribution pi(x|y_obs,tau,beta)
  #the proposed implementation uses a matrix representation; one can
  #also draw the samples using a loop
  v = np.random.normal(0,1,(n[0],ns))
  v0 = np.matlib.repmat( x_map, 1, ns)

  #draw samples
  x_samp = np.linalg.inv(C) @ (np.linalg.inv(C.T) * tau @ K.T @ y_obs + v) # ADD YOUR CODE HERE

  #compute mean
  x_mean = x_samp.mean(axis=1)

  #compute 95% credibility intervals
  # x_quant = cr.getEmpQuant(x_samp.T.conj(), np.array([0.025,0.975])).T.conj()
  x_quant = np.quantile(np.asarray(x_samp), [0.025, 0.975], axis=1)

  #visualize results / data
  fig, ax = plt.subplots(1,2,figsize=(15,5))
  ax[0].plot(s, x_true, color='black', label='true solution')
  ax[0].scatter(s, y_obs, marker='x', color='red', label='observation')
  ax[0].legend()
  ax[1].plot(s,x_true, color='black', label='true solution')
  ax[1].plot(s,x_map, color='red', label='map estimate')
  ax[1].legend()

  plt.figure(figsize=(15,5))
  plt.plot(s, x_true, label='true solution')
  plt.plot(s, x_quant[0, :], label='0.025%')
  plt.plot(s, x_quant[1, :], color='black', label='0.975%')
  plt.plot(s, x_mean, color='red', label='mean')
  plt.legend()
  