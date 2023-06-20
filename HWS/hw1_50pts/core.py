import numpy as np
import matplotlib.pyplot as plt


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
    # fix the random seed generation
    np.random.seed(0)
    eta = np.expand_dims(np.random.normal(0, 1, len(y)), axis=1)
    noise = delta * eta

    # perturb data by noise
    ydelta = y + noise
    if return_noise:
        return ydelta, noise
    else:
        return ydelta
    

def tSVD(K, r):
    '''
    TRUNSVD compute truncated SVD
    
    input:
        r   -    target rank
    
    output:
       Ur   -   left singular vectors
       Sr   -   singular values
       Vr   -   right singular vectors
    '''

    # compute singular value decomposition
    U, S, VT = np.linalg.svd(K) # ADD YOUR CODE HERE

    # compute low rank approximation
    Ur = U[:,:r] # ADD YOUR CODE HERE
    Sr = S[:r] # ADD YOUR CODE HERE
    VTr = VT[:r,:] # ADD YOUR CODE HERE

    # truncation (numerical accuracy)
    Sr[Sr < 1e-14] = 1e-14

    return Ur, Sr, VTr


def tSVDTH(K, alpha):
    '''
    TSVDTH compute truncated SVD
    
    input:
       r    -   target rank
    
    output:
       Ur   -   left singular vectors (truncated)
       Sr   -   singular values (truncated)
       Vr   -   right singular vectors (truncated)
    '''

    # compute SVD
    U, S, VT = np.linalg.svd(K)

    # find id for cutting SVD
    # if there is no value smaller than alpha, use all
    # singular vectors and singular values

    # compute low rank approximation
    Ur = U[:, S>alpha] # ADD YOUR CODE HERE
    Sr = S[S>alpha] # ADD YOUR CODE HERE
    VTr = VT[S>alpha, :] # ADD YOUR CODE HERE

    return Ur, Sr, VTr


def evalLCurve(K, y_delta, solve, alphalist):
    '''
    EVALLCURVE compute and display l-curve
    
    input:
       K         -  matrix operator
       ydelta    -  perturbed data
       solve     -  function handle to solve for x_alpha as a function of the regularization parameter alpha
       alphalist -  list of trial regularization parameters
    '''

    # problem setup
    m  = len(alphalist)
    res = np.zeros((m, 1))
    reg = np.zeros((m, 1))

    # define matrix vector product based on input
#     if isa( K, 'function_handle' ), matvec = @(x) K(x);
#     else,                           matvec = @(x) K*x;
#     end
    
    for i in range(m):
        # get regularization parameter from list
        alpha = alphalist[i]
        # solve inverse problem for selected alpha
        x_alpha = solve(alpha)

        # compute criteria/values for l-curve
        res[i] = np.linalg.norm(K@x_alpha - y_delta) # ADD YOUR CODE HERE
        reg[i] = np.linalg.norm(x_alpha) # ADD YOUR CODE HERE

        print('alpha={}:  ||r|| = {}  ||x|| = {}'.format(alpha, res[i][0], reg[i][0]))

    # PLOT L-CURVE
    plt.figure(figsize=(15,5))
    plt.plot(res, reg, c='black')
    plt.scatter(res, reg, marker='x')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.xlabel(r'$||Kx_{\alpha} - y^{\delta}||_2$', size=14)
    plt.ylabel(r'$||x_{\alpha}||_2$', size=14)
    plt.title('L-curve', size=14)
    

def evalDisPrinc(K, y_delta, solve, alphalist, delta, return_res=False):
    '''
    EVALDISPRINC evaluate discrepancy principle
    
    input:
       K         -  matrix operator
       ydelta    -  perturbed data
       solve     -  function handle to solve for x_alpha (for different values of alpha)
       delta     -  amplitude of perturbation
       alphalist -  list of regularization parameters to be considered
    
    output:
       alpha     -  optimal regularization parameter
    '''

    m = len(alphalist)
    res = np.zeros((m, 1))

    # sort list of trial parameters
    alphalist[::-1].sort() #sort(alphalist, 'descend')

    alpha = -1

    # apply mor
    for i in range(m):
        alpha_trial = alphalist[i]
        x_alpha = solve(alpha_trial)

        # compute error
        err = np.linalg.norm(K@x_alpha - y_delta) # ADD YOUR CODE HERE
        res[i] = err

        # store optimal regularization paramter
        if (err < delta) and (alpha == -1): #if (err < delta && alpha == -1):
            alpha = alpha_trial
            print('err = {} <= {} = delta'.format(err, delta))
            print('optimal regularization parameter:', alpha)
            if not return_res:
                return alpha

    plt.figure(figsize=(15, 5))
    plt.scatter(alphalist, res, marker='x')
    plt.plot(alphalist, res, color='black')
    plt.plot(alphalist, delta*np.ones((m, 1)), color='red')
    plt.xscale('log', base=10)
    plt.ylabel(r'$||Kx_{\alpha} - y^{\delta}||_2$', size=19)
    plt.xlabel(r'$\alpha$', size=19)
    plt.title('Discrepancy principle', size=19)
    plt.grid()
    return alpha, res