import core as cr
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.interpolate import make_interp_spline
from scipy.sparse import linalg, diags, identity, vstack, csr_matrix
from scipy.linalg import sqrtm, pinv

######################################################## TASK 1 ########################################################
########################################################################################################################


def scPrecMatPD():

  n = [256]

  # get 1D laplacian opeator with dirichlet boundary conditions
  L = cr.getLapMat(n=n, dim=1)
  t = np.linspace(0, 1, n[0])

  # compute spectrum of laplacian operator
  U, S, V = np.linalg.svd(L) #full(L) what's in python?
  s = np.diag(S)
  print('smallest singular value {}'.format(min(S)))

  # visulaize spectrum of laplacian operator
  f, ax = plt.subplots(1, 5, figsize=(15, 5))
  ax[0].plot(t, S)
  ax[0].set_yscale('log')
  ax[0].set_title('singular values')
  ax[1].plot(t, U[:, 128])
  ax[1].set_title('singular vector 128')
  ax[2].plot(t, U[:, 225])
  ax[2].set_title('singular vector 225')
  ax[3].plot(t, U[:, 250])
  ax[3].set_title('singular vector 250')
  ax[4].plot(t, U[:, 255])
  ax[4].set_title('singular vector 256')
  plt.tight_layout()

  # check orthogonality
  ntrials = 10
  k = np.random.randint(1, n[0], ntrials).reshape(ntrials, 1)
  j = np.random.randint(1, n[0], 1).reshape(1, 1)
  uj = U[:, j[0][0]]

  for i in range(ntrials):
      uk = U[:, k[i]]
      #print(uk.T.conj().shape)
      #print(uj.shape)
      print('inner product {} {}: {}'.format(j, k[i], uk.T.conj() @ uj))
      print('inner product {} {}: {}'.format(k[i], k[i], uk.T.conj() @ uk))

  # symmetry error: <L*x, L*x> - <L*L*x,x>
  x = np.random.rand(n[0], 1)
  err = ((L@L@x).conj().T @ x - (L@x).conj().T @ (L@x)) / np.linalg.norm(L@x)
  print('error in symmetry: {}'.format(err))


######################################################## TASK 4 ########################################################
########################################################################################################################


def scDrawGMRFDBC1D():
  #draw a realization from Gaussian probability distribution
  #with \bar{x}=0 and covariance matrix C = L^{-1}, where L is the
  #precision matrix (finite difference approximation of the laplacian
  #operator; the implementation is in one and two dimensions

  #problem dimension (256 x 1)
  n = 256

  #get finite difference approximation of laplacian operator (i.e.,
  #precision matrix L)
  L = cr.getLapMat([n, 1], 1)

  #compute cholesky decomposition
  C = np.linalg.cholesky(L) 

  #number of draws
  ndraws = 6
  plt.figure(figsize=(15,5))
  s = np.linspace(0, 1, n)

  for i in range(ndraws):

      #draw x from normal distribution
      w = np.random.normal(size=(n,1)) # ADD YOUR CODE
      x = np.linalg.inv(C) @ w # ADD YOUR CODE

      #plot drawn x
      plt.plot(s, x)


def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
  
  n = A.shape[0]
  LU = linalg.splu(A, diag_pivot_thresh=0) # sparse LU decomposition
  
  if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
    return LU.L.dot( diags(LU.U.diagonal()**0.5) )
  else:
    sys.exit('The matrix is not positive definite')


def scDrawGMRFDBC2D():
  #draw a realization from Gaussian probability distribution
  #with \bar{x}=0 and covariance matrix C = L^{-1}, where L is the
  #precision matrix (finite difference approximation of the laplacian
  #operator; the implementation is in one and two dimensions

  # problem dimension (256 x 256)
  n = 256

  # get finite difference approximation of laplacian operator (i.e.,
  # precision matrix L)
  L = cr.getLapMat([n, n], 2)
  eigenvalues, V = linalg.eigs(L[0], k=24)
  LAMBDA = np.diag(eigenvalues)

  #number of draws
  ndraws = 7
  f = plt.figure(figsize=(20, 8))

  for i in range(1, ndraws):

      #draw x from normal distribution
      w = np.random.normal(size=(24, 1)) # ADD YOUR CODE
      x = V @ pinv(LAMBDA**(1/2)) @ w # ADD YOUR CODE

      ax = f.add_subplot(2, 3, i)
      ax.imshow(x.reshape(n, n).real, extent=[0,1,0,1], cmap='gray')
      ax.set_aspect('equal', adjustable='box')
  plt.tight_layout()

######################################################## TASK 5 ########################################################
########################################################################################################################


def scDrawIGMRFNBC1D():
  #1D independent (but not identically distributed) increment case
  #with x_{i+1}-x_i ~ N(0,w_i), where w_n/2 = 0.05 and w_i=1 otherwise
  #that is, we implement sampling from an IGMRF  field in a one-dimensional
  #setting; the precision matrix defines a probability distribution that
  #models a prior that preserves rapid changes (jumps) in the to be
  #reconstructed signal; the precision matrix does not allow for a cholesky
  #factorization


  n = 128 # number of points (discretization)
  ndraws = 6 # number of draws
  eps = np.finfo(float).eps # perturbation factor

  #get gradient operator with neumann BC
  D = cr.getGradMat([n], 1)

  #get normaly distributed random vector v
  v = np.random.normal(size=(n-1, ndraws))

  #compute square of weight matrix
  W = np.eye(n-1)
  W[n//2, n//2] = np.sqrt(0.0025)

  #compute matrix \tilde{C}
  C_tilde = W @ D # ADD YOUR CODE HERE
  # M = # ADD YOUR CODE HERE

  #compute cholesky factorization of \tilde{C}^T\tilde{C} + eps*I
  denom = C_tilde.T@C_tilde + eps*np.eye(n) # ADD YOUR CODE HERE

  #compute samples
  x =  np.linalg.pinv(denom) @ C_tilde.T @ v # ADD YOUR CODE HERE

  #visualize zamples
  plt.figure(figsize=(15,5))
  s = np.linspace(0, 1, n)
  plt.plot(s, x)


def getW(n, D1, D2):
  '''
  GETW compute weight matrix
  
  input:
    n     number of grid points
  
  output:
    W     weight matrix
  '''

  #compute weight matrix W
  #compute 2 norm centered at (0.5,0.5); domain is [0,1] x [0,1]
  h = 1/(n+1) # compute spatial step size
  x = np.arange(h, 1-h+h, h) #h:h:1-h # compute 1D axis
  x1, x2 = np.meshgrid(x, x) # compute 2D mesh

  r = (x1 - 0.5)**2 + (x2 - 0.5)**2
  cid = ( r < 0.1 ) # indicator function on a circle
  grcid = np.sqrt( (D1@cid.reshape(-1,1))**2 + (D2@cid.reshape(-1,1))**2 ) # compute boundary of circle

  w = np.ones((n*(n-1), 1)) # compute diagonal of weight matrix
  w[grcid > 0] = np.sqrt(0.0025) # set boundary of circle to 0.05

  W = diags([np.squeeze(w)], [0], shape=(n*(n-1), n*(n-1)) )

  return W


def scDrawIGMRFNBC2D():
  #2D independent (but not identically distributed) increment case
  #x_{i+1,j}-x_{i,j} ~ N(0,w_ij) and x_{i,j+1}-x_{i,j} ~ N(0,w_ij),
  #where w_ij=0.05 on the boundary of a circle and w_ij=1 otherwise

  #that is, we implement sampling from an IGMRF field in a two-dimensional
  #setting; the precision matrix defines a probability distribution that
  #models a prior that preserves rapid changes (jumps) in the to be
  #reconstructed signal; the precision matrix does not allow for a cholesky
  #factorization

  n = 128 # number of points (discretization)
  ndraws = 7 # number of draws
  eps = np.finfo(float).eps # perturbation factor

  #get gradient operator with neumann BC
  D, D1, D2 = cr.getGradMat([n,n], 2)

  #compute W
  W = getW(n, D1, D2)

  #compute [WD_1; WD_2]
  C_tilde = vstack((W@D1, W@D2)) # ADD YOUR CODE HERE

  #compute matrix \tilde{D}^T\tilde{D} + eps*I
  denom = C_tilde.T.dot(C_tilde) + eps*identity(n*n) # ADD YOUR CODE HERE

  f = plt.figure(figsize=(20, 8))

  for i in range(1, ndraws):
      #get normaly distributed random vector v
      v = csr_matrix(np.random.normal(size=(2*n*(n-1), 1))) # ADD YOUR CODE HERE
      u, sigma, vt = linalg.svds(denom)

      #draw x
      x = vt.T @ np.linalg.pinv(np.diag(sigma)) @ u.T @ C_tilde.T @ v # ADD YOUR CODE HERE

      ax = f.add_subplot(2, 3, i)
      ax.imshow(x.reshape(n, n), extent=[0,1,0,1], cmap='gray')
      ax.set_aspect('equal', adjustable='box')
  plt.tight_layout()

