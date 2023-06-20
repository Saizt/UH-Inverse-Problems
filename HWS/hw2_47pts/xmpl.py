import core as cr
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg

def exSolLSCG():    

    s = -3 # order of smallest eigenvalue (lmin = 10^s), s < 0
    n = 512 # size of SPD matrix
    K = cr.getSPDMat(n, s) # compute SPD matrix

    # get condition number
    c = np.linalg.cond(K)
    print('condition number of K:', c)

    # compute random vector xtrue
    xtrue = np.random.normal(0, 1, (n, 1))

    # compute RHS/data y given xtrue
    y = K @ xtrue

    # solve linear system Kx = y
    xsol1 = cg(K, y, tol=1e-6)[0] #pcg(K, y, 1e-6)
    xsol2 = cr.runCG(K, y, 1e-6, n)

    # compute relative error between true solution and numerical solution
    err1 = np.linalg.norm(xsol1-xtrue) / np.linalg.norm(xtrue)
    print('xsol1 relative error:', err1)
    err2 = np.linalg.norm(xsol2-xtrue) / np.linalg.norm(xtrue)
    print('xsol2 relative error:', err2)

    # plot xtrue versus numerical solutions
    plt.figure(figsize=(15, 5))
    plt.plot(xsol2, c='r', marker='o', label='xsol2')
    plt.plot(xtrue, c='k', marker='x', label='x_true')
    plt.plot(xsol1, label='xsol1')
    plt.legend()

    # display matrix K
    f = plt.figure(figsize=(15, 5))
    ax = f.add_subplot(1, 2, 1)
    im1 = ax.imshow(K, extent=[0,1,0,1])
    plt.colorbar(im1, ax=ax)
    ax.set_title('K')

    ax = f.add_subplot(1, 2, 2)
    im1 = ax.imshow(abs(np.log(K)), extent=[0,1,0,1])
    plt.colorbar(im1, ax=ax)
    ax.set_title('|ln(K)|')
    plt.tight_layout()