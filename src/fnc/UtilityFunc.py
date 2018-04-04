def DefSystem(np):

    # A \in \mathbb{R}^{n \dot n \dot r} where n in the dimension of the state and r is the number of regions
    # basically A[:,:,1] is the system dynamics in region 1
    A = [np.array([[1.0, 1.0],
                   [0., 1.0]]),
         np.array([[0.9, 0.7],
                   [0. , 0.9]]),
         np.array([[0.6, 0.4],
                   [0., 0.7]])]

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
                        [-0.075,  1],
                        [-0.075, -1],
                        [    2, -1]]),
              np.array([[-0.075,  1],
                        [-0.075, -1],
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

def PlotRegions(Vertex, plt, np, Vrep, Hrep, x):

    plt.plot(np.squeeze(Vertex[0])[:,0], np.squeeze(Vertex[0])[:,1], "r*", marker="*", markersize=15)
    plt.plot(np.squeeze(Vertex[1])[:,0], np.squeeze(Vertex[1])[:,1], 'bo', marker="o", markersize=10)
    plt.plot(np.squeeze(Vertex[2])[:,0], np.squeeze(Vertex[2])[:,1], 'rs', marker="s", markersize=5)
    plt.plot(x[0,:], x[1,:], '-ro')

    plt.xlim([-2.5, 2.5])
    plt.ylim([-1, 4.5])

    plt.show()


    return 1


def PlotRegionsResult(Vertex, plt, np, Vrep, Hrep, x, x_SteadyState):
    plt.plot(np.squeeze(Vertex[0])[:, 0], np.squeeze(Vertex[0])[:, 1], "r*", marker="*", markersize=15)
    plt.plot(np.squeeze(Vertex[1])[:, 0], np.squeeze(Vertex[1])[:, 1], 'bo', marker="o", markersize=10)
    plt.plot(np.squeeze(Vertex[2])[:, 0], np.squeeze(Vertex[2])[:, 1], 'rs', marker="s", markersize=5)
    plt.plot(x[0, :], x[1, :], '-ro')
    plt.plot(x_SteadyState[0, :], x_SteadyState[1, :], '-bo')

    plt.xlim([-2.5, 2.5])
    plt.ylim([-1, 4.5])

    plt.show()

    return 1