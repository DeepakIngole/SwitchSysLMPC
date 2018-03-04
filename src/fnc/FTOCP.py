def FTOCP(M, G, E, x0, optimize, np, z0):

    def cost(z):
        return 0.5 * ( np.dot(z.T, np.dot(M, z)) )

    def jac(z):
        return np.dot(z.T, M)

    cons = {'type':'eq',
            'fun':lambda z: np.dot(E, x0)- np.dot(G,z), # E*x0 - Gz = 0
            'jac':lambda z: -G}
    # cons = {'type':'ineq',
    #         'fun':lambda x: b - np.dot(A,x), # b - Ax => 0
    #         'jac':lambda x: -A}

    opt = {'disp':True}

    res_cons = optimize.minimize(cost, z0, jac=jac,constraints=cons, method='SLSQP', options=opt)

    return res_cons

def BuildMatEqConst(A ,B ,N ,n ,d ,np):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        Gx[np.ix_(ind1, ind2x)] = -A[0]

        ind2u = i * d + np.arange(d)
        Gu[np.ix_(ind1, ind2u)] = -B[0]

    G = np.hstack((Gx, Gu))
    E = np.zeros((n * (N + 1), n))
    E[np.arange(n)] = np.eye(n)

    # Given the above matrices the dynamic constrain is give by Gz=Ex(0) where x(0) is the measurement
    # For sanity check plot the matrices
    print("Print Gx")
    print(Gx)
    print("Print Gu")
    print(Gu)
    print("Print G")
    print(G)
    print("Print E")
    print(E)

    return G, E

def BuildMatCost(Q, R, P, N, linalg):
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M = linalg.block_diag(Mx, P, Mu)

    # For sanity check
    print("Cost Matrix")
    print(M)

    return M

def GetPred(Solution,n,d,N, np):
    xPred = np.squeeze(np.transpose(np.reshape((Solution.x[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution.x[n*(N+1)+np.arange(d*N)]),(d, N))))

    return xPred, uPred