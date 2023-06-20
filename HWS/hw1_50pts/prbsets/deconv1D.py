import sys
import numpy as np
sys.path.append('../')
import core as cr
import matplotlib.pyplot as plt


######################################################## TASK 1 ########################################################
########################################################################################################################


def getKernel1D(n, tau=0.03, kx=False):
    '''
    GETKERNEL1D function to get discrete convolution operator/kernel matrix
    
    input:
        n    -   number of points
        tau  -   bandwidth of kernel
    
    output:
        K    -   kernel matrix
    '''
    
    # spatial step size (domain is [0,1])
    h = 1/n 
    h2 = h*h
    
    # compute all-to-all distance
    y = np.expand_dims(np.arange(0.5, n+0.5), axis=0) - np.expand_dims(np.arange(0.5, n+0.5), axis=0).conj().T #y = (0.5:n-0.5) - (0.5:n-0.5)'
    
    # constants for kernel
    c = 1 / (np.sqrt(2*np.pi) * tau)
    d = h2 / (2 * tau**2)
    
    # function handle to construct discrete kernel matrix
    ker = lambda x: c * np.exp(-d * x**2) #ker = @(x) c*exp(-d*(x.^2));
    
    # discrete convolution matrix / kernel matrix
    K = h * ker(y)
    
    # if 1D kernel is requested by user, provide it
    if kx:
        kx = ker( np.arange(-(n-0.5), n+0.5, 2) )
        return K, kx
    else:
        return K
    
    
######################################################## TASK 2a ########################################################
#########################################################################################################################    
  
    
def scCompKerSVD1D():
    n = 256 # number of points
    w = np.linspace(0, 1, n) # domain

    # get disrete convolution operator
    K = getKernel1D(n, 0.03)

    # compute the singular value decomposition of the matrix K
    U, S, VT = np.linalg.svd(K) # ADD YOUR CODE HERE

    # for visualization only (account for numerical accuracy)
    S[S < 1e-14] = 1e-14

    v01 = VT[0] # ADD YOUR CODE HERE
    v05 = VT[4] # ADD YOUR CODE HERE
    v10 = VT[9] # ADD YOUR CODE HERE
    v20 = VT[19] # ADD YOUR CODE HERE

    # visualize singular values
    plt.figure(figsize=(15,5))
    plt.plot(range(n), S)
    plt.yscale('log')
    plt.title('singular values')

    # visualize singular vectors
    f, ax = plt.subplots(1, 4, figsize=(15,5))
    ax[0].plot(w, v01)
    ax[0].set_title(r'$v_1$ interpreter', size=19)
    ax[1].plot(w, v05)
    ax[1].set_title(r'$v_5$ interpreter', size=19)
    ax[2].plot(w, v10)
    ax[2].set_title(r'$v_{10}$ interpreter', size=19)
    ax[3].plot(w, v20)
    ax[3].set_title(r'$v_{20}$ interpreter', size=19)
    plt.tight_layout()

    
######################################################## TASK 2b ########################################################
#########################################################################################################################
    
    
def scTSVDK1D():
    # implementation of truncated SVD and its analysis
    # for one-dimensional deconvolution problem

    # number of grid points
    n = 256
    s = np.linspace(0, 1, n)
    tau = 0.03

    # get disrete convolution operator
    K = getKernel1D(n, tau)

    # compute truncated SVD for target ranks {5, 10, 50}
    r = [5, 10, 50]
    U1, S1, VT1 = cr.tSVD(K, r[0])
    U2, S2, VT2 = cr.tSVD(K, r[1])
    U3, S3, VT3 = cr.tSVD(K, r[2])

    # construct operator from low rank matrices
    K1 = U1@np.diag(S1)@VT1
    K2 = U2@np.diag(S2)@VT2
    K3 = U3@np.diag(S3)@VT3

    # display low rank approximations of matrix K
    f, ax = plt.subplots(1, 4, figsize=(15,3))
    im1 = ax[0].imshow(K1, extent=[0,1,0,1])
    plt.colorbar(im1, ax=ax[0])
    #axis square
    ax[0].set_title(r'$K_5$')
    im2 = ax[1].imshow(K2, extent=[0,1,0,1])
    plt.colorbar(im2, ax=ax[1])
    #axis square
    ax[1].set_title(r'$K_{10}$')
    im3 = ax[2].imshow(K3, extent=[0,1,0,1])
    plt.colorbar(im3, ax=ax[2])
    #axis square
    ax[2].set_title(r'$K_{50}$')
    im4 = ax[3].imshow(K, extent=[0,1,0,1])
    plt.colorbar(im4, ax=ax[3])
    #axis square
    ax[3].set_title('K')
    plt.tight_layout()

    R1 = np.abs(K - K1) # ADD YOUR CODE HERE
    R2 = np.abs(K - K2) # ADD YOUR CODE HERE
    R3 = np.abs(K - K3) # ADD YOUR CODE HERE

    f, ax = plt.subplots(1, 3, figsize=(15,4))
    im1 = ax[0].imshow(R1, extent=[0,1,0,1])
    plt.colorbar(im1, ax=ax[0])
    #axis square
    ax[0].set_title(r'$R_5$')
    im2 = ax[1].imshow(R2, extent=[0,1,0,1])
    plt.colorbar(im2, ax=ax[1])
    #axis square
    ax[1].set_title(r'$R_{10}$')
    im3 = ax[2].imshow(R3, extent=[0,1,0,1])
    plt.colorbar(im3, ax=ax[2])
    #axis square
    ax[2].set_title(r'$R_{50}$')
    plt.tight_layout()

    # compute relative error between low rank approximations
    # and display error to user
    err1 = np.linalg.norm(K - K1) / np.linalg.norm(K) # ADD YOUR CODE HERE
    err2 = np.linalg.norm(K - K2) / np.linalg.norm(K) # ADD YOUR CODE HERE
    err3 = np.linalg.norm(K - K3) / np.linalg.norm(K) # ADD YOUR CODE HERE
    print(' error (rank  5) = ', err1 )
    print(' error (rank 10) = ', err2 )
    print(' error (rank 50) = ', err3 )
    
    
def PlotResErr():
    #compute Kr matrices, the residual matrices and the errors
    n = 256
    tau = 0.03
    K = getKernel1D(n, tau)
    r = np.arange(1,60)
    recon = lambda UsVT: UsVT[0]@np.diag(UsVT[1])@UsVT[2]
    Kr = [recon(cr.tSVD(K, r)) for r in r]
    errs = [np.linalg.norm(K - Kr[r]) / np.linalg.norm(K) for r in range(len(Kr))]
    
    fig = plt.figure(figsize = (8,4))
    plt.plot(errs)
    plt.title("residual error ||K - Kr|| / ||K||")
    plt.xlabel("r")
    plt.ylabel("error")
    r5 = plt.scatter(4, errs[4])
    r10 = plt.scatter(9, errs[9])
    r50 = plt.scatter(49, errs[49])
    plt.legend((r5, r10, r50), (f"r = 5, err = {errs[4]}", f"r = 10, err = {errs[9]}", f"r = 50, err = {errs[49]}"))
    plt.show()
    

######################################################## TASK 2c ########################################################
#########################################################################################################################    
    
    
def getDeconvSource1D(n, id=3):
    '''
    get source (i.e., x_true) for deconvolution problem
    
    input:
        n   -  number of grid points
    
    output:
        x   -  data set
    '''

    # compute coordinate functions
    h = 1/n
    x = np.expand_dims(np.linspace(h, 1-h, n), axis=0).conj().T

    # compute data
    if id == 1:
        x1 = x > 0.10
        x2 = x < 0.30
        x3 = x > 0.50
        x = x1*x2 + np.sin(4*np.pi*x)*x3
    elif id == 2:
        x1 = x > 0.10
        x2 = x < 0.30
        x3 = x > 0.50
        x = x1*x2 + np.sin(4*np.pi*x)*x3 + 0.2*np.cos(30*np.pi*x)
    elif id == 3:
        i1 = (x > 0.10) & (x < 0.25)
        i2 = (x > 0.30) & (x < 0.35)
        i3 = (x > 0.50) & (x < 1.00)

        x = 0.75*i1 + 0.25*i2 + np.sin(2*np.pi*x)**4 * i3
    else:
        print('id = {} not implemented'.format(id))

    # normalize data
    x = x/np.linalg.norm(x)
    return x
    
    
def scDeconvTSVD1D():

    # number of grid points
    n = 256
    gamma = 50 # signal to noise ratio
    tau = 0.005 # parameter controlling smoothing
    # tau = 0.02

    # get disrete convolution operator
    K = getKernel1D(n, tau)

    # compute condition number
    c = np.linalg.cond(K)
    print('condition number of K:', c)

    # get true data
    x_true = getDeconvSource1D(n)

    # compute right hand side
    y = K @ x_true
    
    # compute noise level as a function of SNR (in dB?)
    delta =  np.linalg.norm(y) / (np.sqrt(n)*gamma) # ADD YOUR CODE HERE

    # perturb right hand side by noise
    y_delta = cr.addNoise(y, delta)

    # define threshold alpha for truncated SVD
    alpha = [0.001, 0.01, 0.1, 0.2, 0.5]

    # compute solution via truncated SVD
    m = len(alpha)
    x_alpha = np.zeros((n, m))

    for i in range(m):
        # compute truncated SVD
        U, S, VT = cr.tSVDTH(K, alpha[i])

        # invert diagonal matrix
        sigma = np.diag(S)
        Sinv = np.linalg.inv(sigma) # ADD YOUR CODE HERE

        # apply truncated SVD
        # x_alpha[:, i] = np.squeeze(np.linalg.inv(U@sigma@VT) @ y_delta) # ADD YOUR CODE HERE
        x_alpha[:, i] = np.squeeze(VT.T@Sinv@U.T @ y_delta) # ADD YOUR CODE HERE

    # plot results and data coordinates
    s = np.linspace(0, 1, n)

    f, ax = plt.subplots(2, 1, figsize=(15, 5))
    ax[0].plot(s, x_true, c='b', label=r'$x_{true}$')
    ax[0].legend()
    ax[1].plot(s, y, c='b', label='y')
    ax[1].plot(s, y_delta, c='orange', label=r'$y^{\delta}$')
    ax[1].legend()


    f, ax = plt.subplots(1, m, figsize=(15, 5))
    for i in range(m):
        ax[i].plot(s, x_alpha[:, i], c='orange', label=r'$x_\alpha$')
        ax[i].plot(s, x_true, c='b', label=r'$x_{true}$')
        ax[i].set_title('alpha = {}'.format(alpha[i]))
        ax[i].legend()
    plt.tight_layout()
     
        
######################################################## TASK 3a ########################################################
#########################################################################################################################


def scDeconvTRegDir1D(alphalist=[1e-12, 1e-3, 1e-1], plot=True):
    # implementation of tikhonov regularization scheme and its analysis
    # for one-dimensional deconvolution problem

    n = 256 # number of grid points
    gamma = 50 # signal to noise ratio
    tau = 0.005 # parameterization of kernel

    # coordinates
    s = np.linspace(0, 1, n)

    # get disrete convolution operator
    K = getKernel1D(n, tau)

    # get true data
    x_true = getDeconvSource1D(n)

    # compute right hand side
    y = K @ x_true

    # compute noise level as a function of snr
    delta = np.linalg.norm(y) / (np.sqrt(n)*gamma) # ADD YOUR CODE HERE

    # perturb right hand side by noise
    y_delta, noise = cr.addNoise(y, delta, return_noise=True)

    # define function handle to solver inverse problem
    solve = lambda alpha: np.linalg.lstsq(K.T@K + alpha*np.eye(n), K.T@y_delta, rcond=None)[0] # ADD YOUR CODE HERE
    #solve = lambda alpha: np.linalg.solve(K.T@K + alpha*np.eye(n), K.T@y_delta) # ADD YOUR CODE HERE

    x = []
    for alpha in alphalist:
        # solve tikhonov system for varying alpha
        x.append(solve(alpha))

    if plot:
        # plot results
        f, ax = plt.subplots(3, 1, figsize=(15, 10))
        ax[0].plot(s, x_true, label=r'$x_{true}$')
        ax[0].plot(s, x[0], label=r'$x$')
        ax[0].set_title(r'$\alpha = {}$'.format(alphalist[0]))
        ax[0].legend()
        ax[1].plot(s, x_true, label=r'$x_{true}$')
        ax[1].plot(s, x[1], label=r'$x_\alpha$') 
        ax[1].set_title(r'$\alpha = {}$'.format(alphalist[1]))
        ax[1].legend()
        ax[2].plot(s, x_true, label=r'$x_{true}$')
        ax[2].plot(s, x[2], label=r'$x_\alpha$')
        ax[2].set_title(r'$\alpha = {}$'.format(alphalist[2]))
        ax[2].legend()
    else:
        return K, y_delta
    
    
######################################################## TASK 3b ########################################################
#########################################################################################################################


#implemented in main notebook


######################################################## TASK 3c ########################################################
#########################################################################################################################
    

def scDeconvTRegMDP1P():
    # implementation of tikhonov regularization scheme and its analysis
    # for one-dimensional deconvolution problem

    n = 256 # number of grid points
    gamma = 50 # signal to noise ratio
    tau = 0.005 # parameterization of kernel

    # coordinates
    s = np.linspace(0, 1, n)

    # get disrete convolution operator
    K = getKernel1D(n, tau)

    # get true data
    x_true = getDeconvSource1D(n)

    # compute right hand side
    y = K @ x_true

    # compute noise level as a function of snr
    delta = np.linalg.norm(y) / (np.sqrt(n)*gamma) # ADD YOUR CODE HERE

    # perturb right hand side by noise
    y_delta, noise = cr.addNoise(y, delta, return_noise=True)

    # define function handle to solver inverse problem
    solve = lambda alpha: np.linalg.lstsq(K.T@K + alpha*np.eye(n), K.T@y_delta, rcond=None)[0] # ADD YOUR CODE HERE

    # define trial regularization parameters
    alpha_list = np.logspace(-5, 0, 20)

    # compute l-curve
    mu = np.linalg.norm(noise)
    alpha, res = cr.evalDisPrinc(K, y_delta, solve, alpha_list, mu, return_res=True) #alphalist should be np.array

    
######################################################## TASK 3d ########################################################
#########################################################################################################################


def scDeconvTRegERR1D():
    # implementation of tikhonov regularization scheme and its analysis
    # for one-dimensional deconvolution problem
    
    n = 256 # number of grid points
    gamma = 50 # signal to noise ratio
    tau = 0.005 # parameterization of kernel

    # coordinates
    s = np.linspace(0, 1, n)

    # get disrete convolution operator
    K = getKernel1D(n, tau)

    # get true data
    x_true = getDeconvSource1D(n, 3)

    # compute right hand side
    y = K @ x_true

    # compute noise level as a function of snr
    delta = np.linalg.norm(y) / (np.sqrt(n)*gamma) # ADD YOUR CODE HERE

    # perturb right hand side by noise
    y_delta, noise = cr.addNoise(y, delta, return_noise=True)

    # define function handle to solver inverse problem
    solve = lambda alpha: np.linalg.lstsq(K.T@K + alpha*np.eye(n), K.T@y_delta, rcond=None)[0] # ADD YOUR CODE HERE

    # define trial regularization parameters
    alpha_list = np.logspace(-5, 0, 20)

    # allocate memory
    m = len(alpha_list)
    relerr = np.zeros((m, 1))

    # compute error between solution x_alpha and x_true
    for i in range(m):
        alpha = alpha_list[i]
        x_alpha = solve(alpha)

        relerr[i] = np.linalg.norm(x_true - x_alpha) / np.linalg.norm(x_true) # ADD YOUR CODE HERE
        print('run {}: error for alpha={}: {}'.format(i, alpha, relerr[i]))
    
    print('\nbest alpha =', alpha_list[np.where(relerr == min(relerr))[0]][0])

    # plot error versus the regularization parameter
    plt.figure(figsize=(15, 5))
    plt.scatter(alpha_list, relerr, marker='x')
    plt.plot(alpha_list, relerr, color='black')
    plt.xscale('log', base=10)
    plt.ylabel(r'$||x_{\alpha} - x_{true}||_2 / ||x_{true}||_2$', size=19)
    plt.xlabel(r'$\alpha$', size=19)
    plt.grid()

    
######################################################## TASK 4a #########################################################
##########################################################################################################################


def getRecoKernel1D(n):
    
    h = 1 / n
    s = np.expand_dims(np.arange(-1 + h, 1, h), axis = 0)
    s = s.conj().T
    
    tau = [.1, .2]
    s1 = s[:n,:]
    s2 = s[n:,:]
    kerL = np.exp(0.5 * -s1 ** 2 / tau[0] ** 2)
    kerR = np.exp(0.5 * -s2 ** 2 / tau[1] ** 2)
    
    ker = np.concatenate((kerL, kerR))
    nn = np.sum(ker, axis = 0) * h
    x_true = ker / nn
    return x_true, s


def getRecoMat1D(n):
    '''
    GETKERNEL1D function to get discrete convolution operator/kernel matrix
    
    input
       n    -   number of points
    
    output
       K    -   kernel matrix
    '''

    # spatial step size (domain is [0,1])
    h = 1 / n
    m = 2*n - 1
    K = np.tril(np.ones((m, m))) * h
    return K
    
    
def scKerRecoTSVD1D():
    
    n = 256
    gamma = 50
    
    #get reco mat
    K = getRecoMat1D(n)
    
    c = np.linalg.cond(K)
    x_true, s = getRecoKernel1D(n)
    
    y = K @ x_true
    delta = np.linalg.norm(y) / (np.sqrt(n) * gamma)
    y_delta, noise = cr.addNoise(y, delta, return_noise=True)
    
    alpha = [0.001, 0.01, 0.05, 0.08, 0.1]
    m = len(alpha)
    x_alpha = np.zeros((2*n-1, m))
    
    for i in range(m):
        U, S, VT = cr.tSVDTH(K, alpha[i])
        sigma = np.diag(S)
        Sinv = np.linalg.inv(sigma)
        #x_alpha = np.linalg.inv((U @ sigma @ Vt)) @ y_delta
        x_alpha[:, i] = np.squeeze(VT.T @ Sinv @ U.T @ y_delta)
    
    f, ax = plt.subplots(1, m, figsize=(15, 5))
    for i in range(m):
        ax[i].plot(x_alpha[:, i], c="orange", label=r'$x_{\alpha}$')
        ax[i].plot(x_true, c="b", label=r'$x_{true}$')
        ax[i].set_title(r'$\alpha = {}$'.format(alpha[i]))
        ax[i].legend()
    
    
######################################################## TASK 4b #########################################################
##########################################################################################################################


def scKerRecoTRegLC1D():
    # implementation of tikhonov regularization scheme and its analysis
    # for one-dimensional deconvolution problem

    n = 256 # number of grid points
    gamma = 50 # signal to noise ratio

    # get reco mat
    K = getRecoMat1D(n)

    # get kernel to be reconstructed
    x_true, s = getRecoKernel1D(n)

    # compute right hand side
    y = K @ x_true

    # compute noise level as a function of snr
    delta = np.linalg.norm(y) / (np.sqrt(n)*gamma) # ADD YOUR CODE HERE

    # perturb right hand side by noise
    y_delta, noise = cr.addNoise(y, delta, return_noise=True)

    # define function handle to solver inverse problem
    solve = lambda alpha: np.linalg.lstsq(K.T@K + alpha*np.eye(2*n-1), K.T@y_delta, rcond=None)[0] # ADD YOUR CODE HERE

    # define trial regularization parameters
    alpha_list = np.logspace(-5, 0, 20)

    # compute l-curve
    cr.evalLCurve(K, y_delta, solve, alpha_list)