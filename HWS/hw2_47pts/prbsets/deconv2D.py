import sys
import numpy as np
sys.path.append('../')
import core as cr
import matplotlib.pyplot as plt
import scipy.io


from matplotlib import cm
from collections import defaultdict


######################################################## TASK 3a ########################################################
########################################################################################################################


def getKernel1D(n, sig):
    # spatial step size (domain is [0,1])
    h = 1/n
    h2 = h*h

    # compute all-to-all distance
    x = np.expand_dims(np.arange(0.5, n+0.5), axis=0) \
    - np.expand_dims(np.arange(0.5, n+0.5), axis=0).conj().T 
    #x = (0.5:n-0.5) - (0.5:n-0.5)';

    # constants for kernel
    c = 1 / (np.sqrt(2*np.pi) * sig)
    d = h2 / (2 * sig**2)

    # function handle to construct discrete kernel matrix
    ker = h * c * np.exp(-d * (x**2))

    return ker


def vizKernel(K):
    #VIZKERNEL visualize kernel
    n = K[0].shape[0]
    m = K[1].shape[0]

    # vizualize kernel matrix
    x11, x12 = np.meshgrid(np.arange(n), np.arange(n)) #probably you need to start from 1 and finish by n (so n+1)
    x21, x22 = np.meshgrid(np.arange(m), np.arange(m))

    # set up a figure twice as wide as it is tall
    fig = plt.figure(figsize=(15, 5))
    
    # =============
    # First subplot
    # =============

    # set up the axes for the first plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(x11, x12, K[0], rstride=1, cstride=1, 
                           cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlim(1, n)
    ax.set_ylim(1, n)
    ax.set_title('K{1}')
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # ==============
    # Second subplot
    # ==============    
    
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(x21, x22, K[1], rstride=1, cstride=1, 
                          cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_xlim(1, m)
    ax.set_ylim(1, m)
    ax.set_title('K{2}')
    fig.colorbar(surf, shrink=0.5, aspect=10)
    
    
def getKernel2D(n, tau=[0.02, 0.01], dbg=False):
    '''
    GETKERNEL1D function to get discrete convolution operator/kernel matrix
    
    input
       n    -   number of points
       dbg  -   flag to enable debug mode (optional)
    
    output
       K    -   kernel matrix
    '''
    
    K = defaultdict(int)
    K[0] = getKernel1D(n, tau[0])
    K[1] = getKernel1D(n, tau[1])

    if dbg:
        vizKernel(K)

    return K


def scDeconvSVD2D():
    # visualize SVD of kernel matrices; the right singular vector of
    # K2 \otimes K1 (i.e., the columns of V1 \otimes V2) can be expressed
    # as vec(v1i v2j^T), where v1i is the i-th column of V1 and v2j is the
    # j-th column of V2

    # problem dimension
    n = 256

    # get 2D kernel matrix
    K = getKernel2D(n)

    # compute SVD of convolution operators
    U1, S1, VT1 = np.linalg.svd(K[0])
    U2, S2, VT2 = np.linalg.svd(K[1]) # ADD YOUR CODE HERE

    # compute outer product 
    SST = np.kron(np.expand_dims(S1, axis=0), np.expand_dims(S2, axis=0).T) # ADD YOUR CODE HERE

    # visualize outer product (log scale)
    f = plt.figure(figsize=(15, 5))
    ax = f.add_subplot(1, 2, 1)
    im1 = ax.imshow(np.log(SST), extent=[0,1,0,1])
    plt.colorbar(im1, ax=ax)
    ax.set_title('K in 2D')
    
    X, Y = np.meshgrid(range(256), range(256)) #shapes of x, y of SST
    ax = f.add_subplot(1, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, SST, rstride=1, cstride=1, 
                           cmap=cm.coolwarm, linewidth=0, antialiased=True)
    ax.set_title('K in 3D')
    f.colorbar(surf, shrink=0.5, aspect=10)
    plt.tight_layout()

    # plot right singular vectors of K2 \otimes K1 based on v1i v2j^T for
    # i,j = 1,4,16
    id = [0, 3, 15]
    m = len(id)


    f = plt.figure(figsize=(10, 8))
    k = 1
    for i in range(m):
      for j in range(m):
        v1 = np.expand_dims(VT1.T[:, id[i]], axis=1) # ADD YOUR CODE HERE
        v2 = np.expand_dims(VT2.T[:, id[j]], axis=1) # ADD YOUR CODE HERE
        # print(v1.shape, v2.shape)
        # plot right singular vector
        ax = f.add_subplot(3, 3, k)
        im = ax.imshow(v1 @ v2.conj().T, extent=[0, 1, 0, 1])
        plt.colorbar(im, ax=ax)
        ax.set_title('i,j={},{}, sing val = {}'.format(id[i], id[j], round(SST[id[i]][id[j]], 2) ))
        k += 1
    plt.tight_layout()
  
    
######################################################## TASK 3b ########################################################
#########################################################################################################################    
 
    
def scDeconvTRegDirSVD2D():
    # direct solver for optimality conditions of thikhonov regularized solution
    # based on SVD of 1D convolution operators

    gamma = 50 # noise perturbation

    # load data
    data = scipy.io.loadmat('./data/satellite-256x256.mat')
    X_true = data['x_true']

    # regularization parameter
    alpha = 1e-3

    # get problem dimensions
    n = X_true.shape[0]
    assert n==X_true.shape[1]

    # get 2D kernel matrix
    K = getKernel2D(n)

    # blur source: first, K{1} is applied to each column of x,
    # then K{2} is applied to each row of x

    Y = (K[1] @ (K[0] @ X_true).conj().T).conj().T

    # compute noise level as a function of snr
    delta = np.linalg.norm(Y) / (gamma * np.sqrt(n*n))

    # perturb data / add noise
    Y_delta = cr.addNoise(Y, delta)

    # compute SVD of convolution operators
    U1, S1, VT1 = np.linalg.svd(K[0]) # ADD YOUR CODE HERE
    U2, S2, VT2 = np.linalg.svd(K[1])

    # compute solution based on SVD
    SST = np.kron(np.expand_dims(S1, axis=0), np.expand_dims(S2, axis=0).T)
    X_alpha = VT1.T @ \
              ((SST / (SST**2 + alpha*np.ones((n,n)) ) ) * \
              (U1.T @ Y @ U2) ) @ VT2 # ADD YOUR CODE HERE

    # compute least squares solution
    X_ls = np.linalg.pinv(K[0], rcond=alpha) @ Y @ np.linalg.pinv(K[1].T, rcond=alpha) # ADD YOUR CODE HERE

    # visualize the results
    t1 = np.linspace(0, 1, n)
    t2 = np.linspace(0, 1, n)
    t1, t2 = np.meshgrid(t1, t2)

    f = plt.figure(figsize=(15, 8))
    ax = f.add_subplot(2, 3, 1)
    im1 = ax.imshow(X_true, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('X_true')
    ax.axis('off')

    ax = f.add_subplot(2, 3, 2)
    im1 = ax.imshow(Y, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('Y')
    ax.axis('off')

    ax = f.add_subplot(2, 3, 3)
    im1 = ax.imshow(Y_delta, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('Y_delta')
    ax.axis('off')

    ax = f.add_subplot(2, 3, 4)
    im1 = ax.imshow(X_alpha, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('X_alpha')
    ax.axis('off')

    ax = f.add_subplot(2, 3, 5)
    im1 = ax.imshow(abs(X_alpha - X_true), extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('abs(X_alpha - X_true)')
    ax.axis('off')

    ax = f.add_subplot(2, 3, 6)
    im1 = ax.imshow(X_ls, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('X_ls')
    ax.axis('off')

    plt.tight_layout()

    
######################################################## TASK 4a #########################################################
##########################################################################################################################


#implemented in the core.py and xmpl.py
    
    
######################################################## TASK 4b #########################################################
##########################################################################################################################


def evalMatVec(x, K, alpha):
    # apply (K^T K + alpha*id) to vector
    # map vector (lexicographical ordering) back to matrix

    X = x.reshape(K[0].shape[0], K[1].shape[0])

    y = K[0].conj().T @ (K[0] @ X @ K[1]) @ K[1].conj().T + alpha * X # ADD YOUR CODE HERE

    return y.reshape(-1,1)

def scDeconvTRegCGMF2D():
    # matrix free implementation for solution of 2D deconvolution problem;
    # we use a CG method to solve the optimality conditions
    # the matvec is given by a matrix-product that involves the 1D convolution
    # operators along each spatial dimension

    gamma = 50  # perturbation
    data = scipy.io.loadmat('./data/satellite-256x256.mat')
    X_true = data['x_true']

    tol = 1e-3 # tolerance for PCG
    maxit = 1000 # number of max iterations for PCG

    # get problem dimensions
    n = X_true.shape[0]
    assert n==X_true.shape[1]

    # get 2D kernel matrix
    K = getKernel2D(n)

    # blur source: first, K{1} is applied to each column of x,
    # then K{2} is applied to each row of x
    Y = K[0] @ X_true @ K[1].conj().T # ADD YOUR CODE HERE

    # compute noise level as a function of snr
    delta = np.linalg.norm(Y) / (gamma * np.sqrt(n*n))

    # perturb data / add noise
    Y_delta = cr.addNoise(Y, delta)

    # solve optimality system (K'*K + alpha*I) x_alpha = (K'*y_delta)
    # we use an iterative solver based on PCG, since the kernel matrix
    # would be too large to store and form; PCG allows us to invert
    # the system only through the knowledge of the action of the matrix
    # on a vector (i.e., an expression for the matrix vector product,
    # a.k.a, the "matvec)

    # compute right hand side K'y (note: K is a summetrix matrix)
    rhs = K[0].conj().T @ Y_delta @ K[1] # ADD YOUR CODE HERE

    # function handle for matrix vector product (function implemented below)
    matvec = lambda x: evalMatVec(x, K, alpha)

    # define regularization parameter
    alpha = 1e-3

    # solve optimality conditions using CG
    x_alpha = cr.runCG(matvec, rhs.reshape(-1,1), tol, maxit) # ADD YOUR CODE HERE

    # map solution back to matrix shape (for visualization)
    X_alpha = x_alpha.reshape(n,n)


    # visualize the results
    t1 = np.linspace(0, 1, n)
    t2 = np.linspace(0, 1, n)
    t1, t2 = np.meshgrid(t1, t2)

    f = plt.figure(figsize=(20, 6))
    ax = f.add_subplot(1, 3, 1)
    im1 = ax.imshow(X_true, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('X_true')
    ax.axis('off')

    ax = f.add_subplot(1, 3, 2)
    im1 = ax.imshow(Y, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('Y')
    ax.axis('off')

    ax = f.add_subplot(1, 3, 3)
    im1 = ax.imshow(Y_delta, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('Y_delta')
    ax.axis('off')

    plt.tight_layout()

    f = plt.figure(figsize=(15, 5))
    ax = f.add_subplot(1, 2, 1)
    im1 = ax.imshow(X_alpha, extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('X_alpha')
    ax.axis('off')

    ax = f.add_subplot(1, 2, 2)
    im1 = ax.imshow(abs(X_alpha - X_true), extent=[0,1,0,1], cmap='gray')
    plt.colorbar(im1, ax=ax)
    ax.set_title('abs(X_alpha - X_true)')
    ax.axis('off')

    plt.tight_layout()

