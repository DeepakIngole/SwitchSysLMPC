def LMPC(A, B, x, u, it, SSit, np, M, G, E, F, b, PointSS, SSindex, FTOCP_LMPC, n, d, N, SS, Qfun, linalg, optimize, InitialGuess, GetPred, time):
    norm = 100

    TermCons = np.zeros((n, (N+1)*n + N*d))
    TermCons[:, N*n:(N+1)*n] = np.eye(n)

    G_LMPC = np.vstack( ( G , TermCons) )
    E_LMPC = np.vstack( (E, np.zeros((n,n))) )

    TermPoint = np.zeros( (np.shape(E_LMPC)[0] ) )
    t=0

    CostSingleQP = np.zeros((SSit, PointSS))
    while (norm > 10**(-10)):
        start_time = time.clock()
        for j in range(0, SSit):
            for i in range(0, PointSS):

                TermPoint[ np.shape(TermPoint)[0]-n:np.shape(TermPoint)[0]] = SS[:, SSindex+i,it-j]

                [SolutionOpt, feasible, Cost] = FTOCP_LMPC(M, G_LMPC, E_LMPC, TermPoint, F, b, x[:, t, it+SSit], optimize, np, InitialGuess, linalg)
                CostSingleQP[j,i] = Cost + Qfun[SSindex+i,it-j]


        index= np.unravel_index(CostSingleQP.argmin(), CostSingleQP.shape)
        j_star = index[0]
        i_star = index[1]
        # print("Optimal Terminal Point",SS[:, SSindex+i_star,it-j_star] , "Cost", CostSingleQP)
        # print("Cost", CostSingleQP)
        # print("Index", index, CostSingleQP[j_star,i_star])

        TermPoint[ np.shape(TermPoint)[0]-n:np.shape(TermPoint)[0]] = SS[:, SSindex+i_star,it-j_star]

        [SolutionOpt, feasible, Cost] = FTOCP_LMPC(M, G_LMPC, E_LMPC, TermPoint, F, b, x[:, t, it+SSit], optimize, np, InitialGuess, linalg)
        InitialGuess = SolutionOpt.x
        [xPred, uPred] = GetPred(SolutionOpt, n, d, N, np)

        if feasible == 1:
            u[:, t, it+SSit] = uPred[0]
        else:
            u[:, t, it+SSit] = 10000
            print("ERROR: Optimization Problem Infeasible at time", t, "Norm:", norm, SS[:, SSindex+i_star,it-j_star], CostSingleQP)
            norm = -1000000
            break

        # Apply the input to the system
        x[:, t + 1, it+SSit] = np.dot(A[0], (x[:, t, it+SSit])) + np.dot(B[0], (u[:, t, it+SSit]))
        # print "Solver Time ", time.clock() - start_time, "seconds"


        SSindex = SSindex  + i_star + 1

        # print(xPred, i_star)
        # print(SSindex, np.shape(SS))
        norm = np.dot(x[:, t + 1, it].T, x[:, t + 1, it])
        # print("Norm:", norm, "State:", x[:, t + 1, it])
        t = t+1

    # Now apply the open-loop from last prediciton
    for i in range(1, N):
        u[:, t, it+SSit] = uPred[i]
        # Apply the input to the system
        x[:, t + 1, it+SSit] = np.dot(A[0], (x[:, t, it+SSit])) + np.dot(B[0], (u[:, t, it+SSit]))
        t = t + 1

    return x[:,:,it+SSit], u[:,:,it+SSit], t-1



def BuildMatCostLMPC(Q,R,N,np,linalg):
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M = linalg.block_diag(Mx, np.zeros(np.shape(Q)), Mu)

    # For sanity check
    # print("Cost Matrix")
    # print(M)

    return M


def FTOCP_LMPC(M, G, E, TPoint, F, b, x0, optimize, np, z0, linalg):

    def cost(z):
        return ( np.dot(z.T, np.dot(M, z)) )

    def jac(z):
        return 2 * np.dot(z.T, M)


    cons =({'type':'eq',
            'fun':lambda z:  TPoint + np.dot(E, x0)- np.dot(G,z), # TerminalPoint + E*x0 - Gz = 0
            'jac':lambda z: -G},
           {'type':'ineq',
            'fun':lambda z: b - np.dot(F,z), # b - Ax => 0 PAY ATTENTION HERE DIFFERENT CONVETION FROM USUAL
            'jac':lambda z: -F})

    opt = {'disp':False}

    res_cons = optimize.minimize(cost, z0, jac=jac,constraints=cons, method='SLSQP', options=opt)

    #Need To Check Feasibility
    EqConstrCheck = TPoint + np.dot(E, x0) - np.dot(G,res_cons.x)
    if ( (np.dot(EqConstrCheck, EqConstrCheck) < 1e-8) and ( ((b - np.dot(F,res_cons.x))).all > -1e-8)):
        feasible = 1
        QPcost = cost(res_cons.x)
    else:
        feasible = 0
        QPcost = 10000
        # print("Infeasibility")
        # print( ((np.dot(E, x0) - np.dot(G,res_cons.x)).all < 1e-8) )
        # print(np.dot(E, x0)-np.dot(G,res_cons.x))
        # print( ( ((b - np.dot(F,res_cons.x))).all > -1e-8) )
        # print(b-np.dot(F,res_cons.x))

    return res_cons, feasible, QPcost