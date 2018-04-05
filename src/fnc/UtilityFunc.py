def DefSystem(np):

    # A \in \mathbb{R}^{n \dot n \dot r} where n in the dimension of the state and r is the number of regions
    # basically A[:,:,1] is the system dynamics in region 1
    A = [np.array([[0.8,0.1],
                   [0., 0.8]]),
         np.array([[0.85, 0.15],
                   [0. , 0.85]]),
         np.array([[0.9, 0.2],
                   [0., 0.9]])]

    # B \in \mathbb{R}^{n \dot d \dot r} where n and d are the dimensions of the state and inputs and r is the number of regions
    # basically B[:,:,1] is the B matrix in region 1
    B = [np.array([[0.],
                   [1.]]),
         np.array([[0.],
                   [1.]]),
         np.array([[0.],
                   [1.]])]

    Q = np.eye(2)

    R = np.array(10)

    Q_LMPC = 1*np.eye(2)

    R_LMPC = 1*np.array(1)

    Vertex = [np.array([[    2,  1],
                        [0.075,  1],
                        [0.075, -1],
                        [    2, -1]]),
              np.array([[0.075,  1],
                        [0.075, -1],
                        [-2, -1],
                        [-2,  1]]),
              np.array([[2, 3],
                        [2, 1],
                        [-2, 1],
                        [-2, 3],
                        [-2, 4]])]


    return A, B, Q, R, Q_LMPC, R_LMPC, Vertex


def DefineRegions(Vertex, Vrep, Hrep, np):
    # This function define the matrices F_region and b_region
    NumRegions = len(Vertex) # Number of Regions

    F_region = []
    b_region = []
    for i in range(0, NumRegions):
        Polyhedron_Vrep = Vrep(Vertex[i])
        F_region.append(Polyhedron_Vrep.A)
        b_region.append(Polyhedron_Vrep.b)

    return F_region,b_region

def SysEvolution(x, u, F_region, b_region, np, CurrentRegion, A, B):
    CurrReg = CurrentRegion(x, F_region, b_region, np)
    x_next = np.dot(A[CurrReg],x) + np.dot(B[CurrReg],u)
    return x_next

def CurrentRegion(x, F_region, b_region, np):
    NumRegions = len(F_region) # Number of Regions

    Region = np.inf
    toll = 0*1e-15
    for i in range(0, NumRegions):
        if np.alltrue(np.dot(F_region[i], x) <= b_region[i] + toll*np.ones(b_region[i].shape)):
            Region = i
            break
    if Region == np.inf:
        print "ERROR: ",x," Outside Feasible Region"

    return Region

def PlotRegions(Vertex, plt, np, x):
    plt.plot(np.hstack( ((Vertex[0])[:,0], np.squeeze(Vertex[0])[0,0]) ),
             np.hstack( ((Vertex[0])[:,1], np.squeeze(Vertex[0])[0,1]) ), "-rs")
    plt.plot(np.hstack( ((Vertex[1])[:,0], np.squeeze(Vertex[1])[0,0])),
             np.hstack( ((Vertex[1])[:,1], np.squeeze(Vertex[1])[0,1])), "-ks")
    plt.plot(np.hstack( ((Vertex[2])[:,0], np.squeeze(Vertex[2])[0,0])),
             np.hstack( ((Vertex[2])[:,1], np.squeeze(Vertex[2])[0,1])), "-bs")
    plt.plot(x[0,:], x[1,:], '-ro')

    plt.xlim([-2.5, 2.5])
    plt.ylim([-1, 4.5])

    plt.show()


    return 1


def BuildMatIneqConst(N, n, np, linalg, spmatrix, Fx, bx, SelectReg):

    # Build the matrices for the input constraint in each region. In the region i we want Fx[i]x <= bx[b]
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
    MatFx = np.empty((0, 0))
    bxtot  = np.empty(0)

    for i in range(0, N): # No need to constraint also the terminal point --> go up to N
        MatFx = linalg.block_diag(MatFx, Fx[SelectReg[i]])
        bxtot  = np.append(bxtot, bx[SelectReg[i]])

    NoTerminalConstr = np.zeros((np.shape(MatFx)[0], n))  # No need to constraint also the terminal point
    Fxtot = np.hstack((MatFx, NoTerminalConstr))


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

def BuildMatEqConst(A ,B ,N ,n ,d ,np, spmatrix, SelectedRegions):
    # Buil matrices for optimization (Convention from Chapter 15.2 Borrelli, Bemporad and Morari MPC book)
    # We are going to build our optimization vector z \in \mathbb{R}^((N+1) \dot n \dot N \dot d), note that this vector
    # stucks the predicted trajectory x_{k|t} \forall k = t, \ldots, t+N+1 over the horizon and
    # the predicte input u_{k|t} \forall k = t, \ldots, t+N over the horizon
    Gx = np.eye(n * (N + 1))
    Gu = np.zeros((n * (N + 1), d * (N)))

    for i in range(0, N):
        ind1 = n + i * n + np.arange(n)
        ind2x = i * n + np.arange(n)
        Gx[np.ix_(ind1, ind2x)] = -A[SelectedRegions[i]]

        ind2u = i * d + np.arange(d)
        Gu[np.ix_(ind1, ind2u)] = -B[SelectedRegions[i]]

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