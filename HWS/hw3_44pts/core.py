import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import identity, diags
from scipy.optimize import fminbound
from scipy.interpolate import interp1d
from scipy.sparse import diags, kron, eye, linalg

######################################################## TASK 1, 5 ########################################################
########################################################################################################################


def getLapOp1D(n, h):
  # construct 1D laplacian operator
  dxx = diags([2*np.ones((1,n)), -np.ones((1,n)), -np.ones((1,n))], [0, 1, -1], shape=(n, n)) / (h**2)

  return dxx


def getLapMat(n, dim, h=False):
  '''
  GETLAPMAT get negative laplacian operator (in matrix form)
  for a domain defined by omega = [0,1] or omega = [0,1] x [0,1]
  in the one or two dimensional casee, respectively we assume zero
  dirichlet boundary conditions
  
  input:
    n     number of grid points along each axis
    dim   dimensionality of ambient space
    h     spatial step size (optional; for FD approx)
  
  output:
    L     laplacian operator
  '''

  if h==False:
    h=np.ones((dim, 1))

  # identity matrix
  eye = lambda j: identity(n[j])

  # get 1D laplaican operator
  del_ = lambda j: getLapOp1D(n[j], h[j])

  # implementations for one and two-dimensional ambient space
  if dim==1:
    L = del_(0)
    return L

  elif dim==2:
    L1 = kron(eye(1), del_(0))
    L2 = kron(eye(1), del_(0))
    L = L1 + L2 #L = np.vstack((L1, L2))
    return L, L1, L2


def getGradOp1D(n, h):
  '''
  construct 1D gradient operator (neumann boundary conditions)
  
  input:
    n     number of grid points along each axis
    h     spatial step size (optional; for FD approx)
  
  output:
    dx    one dimensional gradient operator
  '''

  dx = diags([-np.ones((1,n)), np.ones((1,n))], [0, 1], shape=(n-1, n)) / h

  return dx


def getGradMat(n, dim, h=False):
  '''
  GETGRADMAT get gradient matrix
  
  input:
    n     number of grid points along each axis
    dim   dimensionality of ambient space
    h     spatial step size (optional; for FD approx)
  
  output:
    D     gradient operator
  '''

  if h==False:
    h=np.ones((dim, 1))

  # identity matrix
  eye = lambda j: identity(n[j])

  # get 1D laplaican operator
  grd = lambda j: getGradOp1D(n[j], h[j])

  # implementations for one and two-dimensional ambient space
  if dim==1:
    D = grd(0)
    return D

  elif dim==2:
    D1 = kron(eye(1), grd(0))
    D2 = kron(grd(1), eye(0))
    L = D1.T@D1 + D2.T@D2 #D = np.vstack((D1, D2))
    return L, D1, D2


######################################################## TASK 2 ########################################################
########################################################################################################################


def evalGCV(K, L, y_obs):
  '''
  implementation of generalized cross validation (GCV) for tikhonov
  regularized linear inverse problem K*x = y
  
  input:
     K        forward operator
     L        regularization matrix
     y_obs    observed data
  
  output:
     alpha    optimal regularization parameter
  '''

  #get size of vector x
  n = L.shape[0]

  #define function handle to compute regularization matrix such that
  #x_nu = K_nu*y_obs
  K_nu = lambda alpha: np.linalg.inv(K.T@K + alpha*L) @ K.T # ADD YOUR CODE HERE

  #define function handle for tikhonov regularized solution x_nu
  x_nu = lambda alpha: K_nu(alpha) @ y_obs # ADD YOUR CODE HERE

  #residual for tikhonov solution
  res = lambda alpha: K @ x_nu(alpha) - y_obs

  #trace of K*K_nu
  tr  = lambda alpha: np.trace(K @ K_nu(alpha))

  #objective function to be minimized with respect to alpha
  fgcv = lambda alpha: np.linalg.norm(res(alpha))**2 / (n - tr(alpha)) # ADD YOUR CODE HERE

  #compute minimizer using matlab's fminbnd
  alpha = fminbound(fgcv, 0, 1)

  return alpha

