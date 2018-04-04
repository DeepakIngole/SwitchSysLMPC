def FTOCP_CVX(M, G, E, F, b, x0, optimize, np, z0, linalg, qp, spmatrix, matrix):
    q = matrix(0.0, (M.size[0], 1))
    res_cons  = qp(M, q, F, matrix(b), G, E* matrix(x0))
    if res_cons['status'] == 'optimal':
        feasible = 1
    else:
        feasible = 0

    return np.squeeze(res_cons['x']), feasible



def FTOCP(M, G, E, F, b, x0, optimize, np, z0, linalg):

    def cost(z):
        return 0.5 * ( np.dot(z.T, np.dot(M, z)) )

    def jac(z):
        return np.dot(z.T, M)


    cons =({'type':'eq',
            'fun':lambda z: np.dot(E, x0)- np.dot(G,z), # E*x0 - Gz = 0
            'jac':lambda z: -G},
           {'type':'ineq',
            'fun':lambda z: b - np.dot(F,z), # b - Ax => 0 PAY ATTENTION HERE DIFFERENT CONVETION FROM USUAL
            'jac':lambda z: -F})

    opt = {'disp':False}

    res_cons = optimize.minimize(cost, z0, jac=jac,constraints=cons, method='SLSQP', options=opt)

    #Need To Check Feasibility
    EqConstrCheck  = np.dot(E, x0) - np.dot(G,res_cons.x)
    IneqConstCheck = b - np.dot(F,res_cons.x)
    if ( (np.dot(EqConstrCheck, EqConstrCheck) < 1e-8) and ( IneqConstCheck > -1e-8).all() ):
        feasible = 1
    else:
        feasible = 0
        print("Infeasibility")
        print( ((np.dot(E, x0) - np.dot(G,res_cons.x)).all < 1e-8) )
        print(np.dot(E, x0)-np.dot(G,res_cons.x))
        print( ( ((b - np.dot(F,res_cons.x))).all > -1e-8) )
        print(b-np.dot(F,res_cons.x))

    return res_cons.x, feasible

def BuildMatEqConst(A ,B ,N ,n ,d ,np, spmatrix):
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
    # print("Print Gx")
    # print(Gx)
    # print("Print Gu")
    # print(Gu)
    # print("Print G")
    # print(G)
    # print("Print E")
    # print(E)

    G_sparse = spmatrix(G[np.nonzero(G)], np.nonzero(G)[0], np.nonzero(G)[1], G.shape)
    E_sparse = spmatrix(E[np.nonzero(E)], np.nonzero(E)[0], np.nonzero(E)[1], E.shape)
    return G, E, G_sparse, E_sparse

def BuildMatIneqConst(N, n, np, linalg, spmatrix):
    # Buil the matrices for the state constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fx = np.array([[[ 1., 0.],
                    [-1., 0.],
                    [ 0., 1.],
                    [ 0.,-1.]],
                   [[ 1., 0.],
                    [-1., 0.],
                    [ 0., 1.],
                    [ 0.,-1.]]])

    bx = np.array([[[ 2.],
                    [ 2.],
                    [ 1.],
                    [ 1.]],
                   [[ 2.],
                    [ 2.],
                    [ 2.],
                    [ 2.]]])

    # Buil the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
    Fu = np.array([[[ 1.],
                    [-1.]],
                   [[ 1.],
                    [-1.]]])

    bu = np.array([[[ 3.],
                    [ 3.]],
                   [[ 3.],
                    [ 3.]]])

    # Now stuck the constraint matrices to express them in the form Fz<=b. Note that z collects states and inputs
    # Let's start by computing the submatrix of F relates with the state
    rep_a = [Fx[0]] * (N)
    Mat = linalg.block_diag(*rep_a)
    NoTerminalConstr = np.zeros((np.shape(Mat)[0],n)) # No need to constraint also the terminal point
    Fxtot = np.hstack((Mat, NoTerminalConstr))
    bxtot = np.repeat(bx[0], N)


    # Let's start by computing the submatrix of F relates with the input
    rep_b = [Fu[0]] * (N)
    Futot = linalg.block_diag(*rep_b)
    butot = np.repeat(bu[0], N)

    # Let's stack all together
    rFxtot, cFxtot = np.shape(Fxtot)
    rFutot, cFutot = np.shape(Futot)
    Dummy1 = np.hstack( (Fxtot                    , np.zeros((rFxtot,cFutot))))
    Dummy2 = np.hstack( (np.zeros((rFutot,cFxtot)), Futot))
    F = np.vstack( ( Dummy1, Dummy2) )
    b = np.hstack((bxtot, butot))


    F_sparse = spmatrix(F[np.nonzero(F)], np.nonzero(F)[0], np.nonzero(F)[1], F.shape)

    return F, b, F_sparse

def BuildMatCost(Q, R, P, N, linalg, np, spmatrix):
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M = linalg.block_diag(Mx, P, Mu)

    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0], np.nonzero(M)[1], M.shape)

    return M, M_sparse

def GetPred(Solution,n,d,N, np):
    xPred = np.squeeze(np.transpose(np.reshape((Solution[np.arange(n*(N+1))]),(N+1,n))))
    uPred = np.squeeze(np.transpose(np.reshape((Solution[n*(N+1)+np.arange(d*N)]),(d, N))))

    return xPred, uPred