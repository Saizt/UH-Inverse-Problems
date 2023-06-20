import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth


def addNoise(y, delta=1, return_noise=False):
    '''
    ADDNOISE add noise to data
    
    input:
       y      -  noise free data
       delta  -  noise level
    
    output:
       ydelta -  perturbed data
       noise  -  perturbation/noise added to data (delta*eta)
    '''
    
    # compute and normalize noise
    eta = np.random.normal(0, 1, y.shape)
    noise = delta * eta

    # perturb data by noise
    ydelta = y + noise
    if return_noise:
        return ydelta, noise
    else:
        return ydelta

    
def runCG(K, y, tol, maxiter=0):
    '''
    RUNCG execute conjugate gradient method to solve
    linear system of equations of form K*x = y
    
    input:
       K       - n x n matrix or function handle for matvec
       y       - right hand side (must be of size n x 1)
       tol     - tolerance for solver
       maxiter - max iteration count (if not set, defaults to n)
    '''

    # determine whether K is a matrix or a function
    ktype = type(K)

    # setup up according to matrix type
    if ktype is np.ndarray: #if strcmp( ktype, 'matrix' )
        # check inputs for appropriate sizes
        m, n = K.shape
        assert m == n, 'matrix not square'
        assert y.shape == (m,1), 'size mismatch'
        # define matrix vector product
        matvec = lambda x: K @ x
    else:
        m = y.shape[0]
        n = m
        assert y.shape[0] > y.shape[1], 'y is not a colum vector'

        # define matrix vector product
        matvec = lambda x: K(x)

    # set to default
    if maxiter == False:
      maxiter = n

    # set initial guess for CG
    x = np.zeros( (n,1) )

    # main function executing CG
    r = matvec(x) - y
    d = -r
    rsold = r.conj().T @ r

    # main CG loop
    for i in range(maxiter):
        kd = matvec(d)
        dtkd = d.conj().T @ kd
        alpha = rsold / dtkd # ADD YOUR CODE HERE

        # curvature condition; if hessian matrix is negative definite,
        # stop iteration (former step is guaranteed to be a descent
        # direction, so we're doing OK)
        if ( dtkd < 0 ):
            if i == 1: 
              x = d
            print( 'negative curvature detected' );
            break
        x = x + alpha * d # ADD YOUR CODE HERE
        r = r + alpha * kd # ADD YOUR CODE HERE
        rsnew = r.conj().T @ r

        # uncomment to debug
        # fprintf(' i = %4d: PCG residual %e\n', i, rsnew);

        if np.sqrt(rsnew) < tol:
          break

        beta = rsnew / rsold # ADD YOUR CODE HERE
        d = -r + beta * d # ADD YOUR CODE HERE
        rsold = rsnew

    print('CG residual {} at iteration {} of {}'.format(np.sqrt(rsnew), i, maxiter))
    
    return x


def getSPDMat(n, s):
    '''
    CREATESPDMAT creates a positive definite matrix
    
    input
       n       size of n x n matrix
       s       order of smallest eigenvalue, i.e., lmin = 10^s
    '''

    if s >= 0:
        print('s should be in (-infty, 0)')
        s = -6

    # construct SPD matrix
    U = orth(np.random.normal(0, 1, (n, n)))
    d = np.logspace(0, s, n)

    A = U@np.diag(d)@U.conj().T

    return A