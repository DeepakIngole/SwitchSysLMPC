def LMPC(A, B, x, u, it, SSit, np, M_sparse, PointSS, SSindex, FTOCP_LMPC_CVX,
         n, d, N, linalg, GetPred, Parallel, p, partial, spmatrix, qp, matrix,
         SelectReg, BuildMatEqConst, BuildMatEqConst_LMPC, BuildMatIneqConst, F_region, b_region, CurrentRegion, SysEvolution, TotCost,
         plt, Vertex, Steps, NumberPlots, IterationPlot, FTOCP_LMPC_CVX_Cost_Parallel, SwLogic, A_true=None, B_true=None):
    
    if A_true is None: 
        A_true = A
    if B_true is None:
        B_true = B
    # ==================================================================================================================
    # ========================== This functions run the it-th LMPC closed loop iteration ===============================
    # ==================================================================================================================

    # The iteration runs until the predicted trajectory reached the terminal point (i.e. in this example point closer to the origin)
    ReachedTerminalPoint = 0         # Flag to check if the iteration is completed

    # For the selected PointSS points in the latest SSit trajectories a QP is solved
    CostSingleQP  = np.zeros((SSit, PointSS))        # This vector stores the cost of each of the (PointSS * SSit) QPs

    # Initilize subset of SS
    SS_sel   = 10000*np.ones((n, PointSS*SSit))       # This vector will contain PointSS points from each of the last SSit-th trajecotries in SS
    Qfun_sel = 10000*np.ones( PointSS*SSit )          # This vector will contain cost associated with the PointSS points from each of the last SSit-th trajecotries in SS
    CostQP   = 10000*np.ones(PointSS*SSit)            # This vector will contain the cost of each QP that we will solve
    CostQP1  = 10000*np.ones(PointSS*SSit)            # This vector will contain the cost of each QP that we will solve

    # ==================================================================================================================
    # ================ Now initialize the main loop for computing the closed loop trajectory ===========================
    t = 0  # Set time = 0
    while (ReachedTerminalPoint == 0):
        # Loop over the latest SSit trajectories to create a vector of terminal constraints
        for j in range(0, SSit):
            # Note that the time index SSindex indicates the first point that has to be considered from the (it-1-j)-th trajectory in SS
            SS_sel[:,j*PointSS + np.arange(0, PointSS)] = x[0:n, SSindex + np.arange(0, PointSS), it - 1 - j]     # Store the terminal point from the (it - 1 - j)-th iteration in the vector SS_sel
            Qfun_sel[j * PointSS + np.arange(0, PointSS)] = TotCost[SSindex + np.arange(0, PointSS), it - 1 - j]  # Store cost associated with the terminal point from the (it - 1 - j)-th iteration in the vector Qfun_sel

        # At each time step is needed to compute the matrices for the equality and inequality constraint. These are
        # computed going through the subset of SS and analysing to which regions the points belong to. For each point in
        # the subset of SS we compute the matrices for the FTOCP which are appended to the following lists.
        List_SelectReg = []     # This list contains the a vector which indicates the region to which each predicted point is constrainted to
        List_G = []             # This list contains the G matrix for the FTOCP
        List_E = []             # This list contains the E matrix for the FTOCP
        List_F = []             # This list contains the F matrix for the FTOCP
        List_b = []             # This list contains the b vector for the FTOCP
        NumParProc = []         # Number of parallelizable process

        # ==============================================================================================================
        # ================================== Compute the matrices for FTOCP ============================================
        for i in range(0, PointSS * SSit):  # Loop over the latest PointSS*SSit points (Subset of SS)

            if (SS_sel[0, i] >= 10000) or (SwLogic == 0) : # Dummy point used to keep the size of SS_sel constant + Switch logic check
                SelectRegNew = SelectReg
            elif (CurrentRegion(SS_sel[:, i], F_region, b_region, np, 0) == SelectReg[-1]): # Terminal point belong to the same
                                                                                            # region as the terminal predicted point
                SelectRegNew = SelectReg
            else:  # The point in subset of SS belong to a different region.
                SelectRegNew = LastIdea(SelectReg, SS_sel, N, i, CurrentRegion, F_region, b_region, np)

            List_SelectReg.append(SelectRegNew)     # Adding vector indicating constrained regions
            if (i > 0) and (List_SelectReg[i - 1] == List_SelectReg[i]).all(): # If no change --> do not compute assign computed ones
                List_G.append(List_G[len(List_G)-1])
                List_E.append(List_E[len(List_E)-1])
                List_F.append(List_F[len(List_F)-1])
                List_b.append(List_b[len(List_b)-1])
            else:   # New regions ---> Need to compute the matrices
                # Compute matrices
                [G_FTOCP, E_FTOCP, _, _] = BuildMatEqConst(A, B, N, n, d, np, spmatrix,
                                                       SelectRegNew)  # Write the dynamics as equality constraint
                [_, _, _, G_LMPC_sparse, E_LMPC_sparse] = BuildMatEqConst_LMPC(G_FTOCP, E_FTOCP, N, n, d,
                                                                                             np, spmatrix)  # Add the terminal constraint
                [_, b, F_sparse] = BuildMatIneqConst(N, n, np, linalg, spmatrix, F_region, b_region, SelectRegNew)

                List_G.append(G_LMPC_sparse)
                List_E.append(E_LMPC_sparse)
                List_F.append(F_sparse)
                List_b.append(b)

            # The below lines are used for parallel computing. Need to work on this probably not the best way of doing this
            if i == 0:
                NumParProc.append(np.array([0, 0]))
            elif (List_SelectReg[i - 1] == List_SelectReg[i]).all():
                NumParProc[len(NumParProc) - 1][1] = NumParProc[len(NumParProc) - 1][1] + 1
            else:
                NumParProc.append(np.array([i, 0]))
        # ==============================================================================================================

        # ==============================================================================================================
        # =========================================== Solve FTOCP ======================================================
        if Parallel == 0:
            for i in range(0, PointSS*SSit):          # Loop over the latest PointSS*SSit points
                    _, _, CostQP[i] = FTOCP_LMPC_CVX(M_sparse, List_G, List_E, n, SS_sel, List_F, List_b,
                                           x[:, t, it], np, Qfun_sel, qp, matrix, spmatrix, i)

        else:
            for k in range(0,len(NumParProc)): # Loop over the points which have the same region of existance (i.e. each predicted point must belong to the same region)
                G_LMPC_sparse = List_G[NumParProc[k][0]]
                E_LMPC_sparse = List_E[NumParProc[k][0]]
                F_sparse      = List_F[NumParProc[k][0]]
                b             = List_b[NumParProc[k][0]]

                Fun = partial(FTOCP_LMPC_CVX_Cost_Parallel, M_sparse, G_LMPC_sparse, E_LMPC_sparse, n, SS_sel, F_sparse, b,
                                           x[:, t, it], np, Qfun_sel, qp, matrix, spmatrix)
                index = np.arange(NumParProc[k][0], NumParProc[k][0]+NumParProc[k][1]+1)                # Create the index vector

                Res = p.map(Fun, index)                                                                 # Run the process in parallel
                CostQP[NumParProc[k][0]:NumParProc[k][0] + NumParProc[k][1]+1] = np.asarray(Res)
        # ==============================================================================================================

        CostSingleQP = CostQP.reshape(SSit, PointSS)                            # Reshape in the more natural format: (Iteration) x (Time)
        index = np.unravel_index(CostSingleQP.argmin(), CostSingleQP.shape)     # Select the indices (Iteration) x (Time) associated with the minimum cost
        j_star = index[0]
        i_star = index[1]

        IndexSelectReg = CostQP.argmin()
        # Solve the Finite Time Optimal Control Problem (FTOCP): This time get the optimal input and also the predicted trajectory
        [Sol, Feasible, _] = FTOCP_LMPC_CVX(M_sparse, List_G, List_E, n, SS_sel, List_F, List_b,
                                               x[:, t, it], np, Qfun_sel, qp, matrix, spmatrix, IndexSelectReg)


        [xPred, uPred] = GetPred(Sol, n, d, N, np)          # Unpack the predicted trajectory

        # ==============================================================================================================
        # ============================================ PLOT ============================================================
        if (NumberPlots > 0) and (it >= IterationPlot) and (it <= IterationPlot + NumberPlots):
            plt.plot(np.hstack(((Vertex[0])[:, 0], np.squeeze(Vertex[0])[0, 0])),
                     np.hstack(((Vertex[0])[:, 1], np.squeeze(Vertex[0])[0, 1])), "-rs")
            plt.plot(np.hstack(((Vertex[1])[:, 0], np.squeeze(Vertex[1])[0, 0])),
                     np.hstack(((Vertex[1])[:, 1], np.squeeze(Vertex[1])[0, 1])), "-ks")
            plt.plot(np.hstack(((Vertex[2])[:, 0], np.squeeze(Vertex[2])[0, 0])),
                     np.hstack(((Vertex[2])[:, 1], np.squeeze(Vertex[2])[0, 1])), "-bs")

            for i in range(0, it):
                plt.plot(x[0, 0:Steps[i] + 1, i], x[1, 0:Steps[i] + 1, i], '-ro')

            plt.plot(x[0, 0:t + 1, it], x[1, 0:t + 1, it], '-bs')
            plt.plot(xPred[0, 0:N+1], xPred[1, 0:N+1], '-g*')

            plt.xlim([-2.5, 2.5])
            plt.ylim([-1, 4.5])

            plt.show()
        # ==============================================================================================================

        # Apply the best input.
        # First check if the best input was feasible (Note that if the QP is not feasible --> cost set to 10000 --> If best problem not feasible all QP were not feasible)
        if Feasible == 1:
            u[:, t, it] = uPred[0]                 # Extract the best input
        else:
            u[:, t, it] = 10000                                                # If not feasible set input to high number
            print("CostQP", CostQP)
            print("ERROR: Optimization Problem Infeasible at time", t)

            ReachedTerminalPoint = 1                                           # Terminate the loop
            break

        # Apply the input to the system
        x[:, t + 1, it] = SysEvolution(x[:, t, it], u[:, t, it], F_region, b_region, np, CurrentRegion, A_true, B_true)
        # print "Solver Time ", time.clock() - start_time, "seconds"

        # Now check if the terminal point used as terminal constraint in the best QP is the terminal point of our task
        # (i.e. the point closed to the origin for the (it-1-j_star)-th trajectory) This is checked propagating terminal
        # point used as terminal constraint along the SS and checking if the value of the 0-th coordinate is equal to
        # the initialization value
        IndexTermPoint = SSindex + i_star  # Time index of the terminal point

        if x[1, IndexTermPoint+1, it-1-j_star] >= 10000:  # Here we check if the 0-th coordinate the propagated point equals the initilization value
            ReachedTerminalPoint = 1                      # If so, set the flag to 1: the simulation is completed
        else:
            # Compute in which regions of the candidate feasible solution belongs
            for l in range(0,N):
                SelectReg[l] = CurrentRegion(xPred[:,l+1], F_region, b_region, np, 1)            # Shift 0:N
            SelectReg[N] = CurrentRegion(x[:, IndexTermPoint+1, it-1-j_star], F_region, b_region, np, 1) # Compute for point N+1

            # Now update the time index used for picking the first point in the trajectories of SS. This step is crucial
            # to guarantee recursive feasibility. It is needed the propagated point in SS will be used as terminal
            # constraint, so that the shifted solution is feasible for the LMPC.
            SSindex = SSindex + i_star + 1  # Now update the time index used for picking the first point in the trajectories of SS.
                                            # This step is crucial to guarantee recursive feasibility. It is needed the
                                            # propagated point in SS will be used as terminal constraint, so that the shifted solution is feasible for the LMPC.

        # Update time index for the simulation
        t = t+1

    # After that the predicted trajectory has planned to reach the terminal point.
    # Now apply the open-loop from last prediction. This is needed to make sure that the closed-loop trajectory does not terminated further from the origin at each iteration.
    # Here we could have solved few QPs with different time horizons. This issue comes from the fact that we want to mimic an infinite horizon control problem.
    for i in range(1, N):
        u[:, t, it] = uPred[i]                              # Extract the input from the predicted ones
        x[:, t + 1, it] = SysEvolution(x[:, t, it], u[:, t, it], F_region, b_region, np, CurrentRegion, A, B)  # Apply the input to the system
        t = t + 1                                                                                              # Update time index for the simulation

    return x[:,:,it], u[:,:,it], t



def BuildMatCostLMPC(Q, R, N, linalg, np, spmatrix):
    b = [Q] * (N)
    Mx = linalg.block_diag(*b)

    c = [R] * (N)
    Mu = linalg.block_diag(*c)

    M = linalg.block_diag(Mx, np.zeros(np.shape(Q)), Mu)

    # For sanity check
    # print("Cost Matrix")
    # print(M)
    M_sparse = spmatrix(M[np.nonzero(M)], np.nonzero(M)[0], np.nonzero(M)[1], M.shape)

    return M, M_sparse


def FTOCP_LMPC(FTOCP_LMPC_Sol, M, G, E, n, it, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i):
    [Sol, feasible, QPcost] = FTOCP_LMPC_Sol(M, G, E, n, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i)

    return QPcost

def FTOCP_LMPC_Sol(M, G, E, n, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i):
    TermPoint[np.shape(TermPoint)[0] - n:np.shape(TermPoint)[0]] = SS_sel[:, i]  # Update the constant vector for the terminal equality

    def cost(z):
        return ( np.dot(z.T, np.dot(M, z)) )

    def jac(z):
        return 2 * np.dot(z.T, M)

    cons =({'type':'eq',
            'fun':lambda z:  TermPoint + np.dot(E, x0)- np.dot(G,z), # TerminalPoint + E*x0 - Gz = 0
            'jac':lambda z: -G},
           {'type':'ineq',
            'fun':lambda z: b - np.dot(F,z), # b - Ax => 0 PAY ATTENTION HERE DIFFERENT CONVETION FROM USUAL
            'jac':lambda z: -F})

    opt = {'disp':False}

    res_cons = optimize.minimize(cost, z0, jac=jac,constraints=cons, method='SLSQP', options=opt)

    # ====================================================================
    # ===================== Need To Check Feasibility=====================
    # ====================================================================
    # There is a small bug in optimize, whenever SLSQP is used and simple box constraint on individual variables are treated
    # as inequality constraint the solver does not detect infeasibility. The cleanest solution is to check feasibility a posteriori
    EqConstrCheck = TermPoint + np.dot(E, x0) - np.dot(G,res_cons.x)
    IneqConstCheck = b - np.dot(F, res_cons.x)
    if ( (np.dot(EqConstrCheck, EqConstrCheck) < 1e-8) and ( IneqConstCheck > -1e-8).all() ):
        feasible = 1
        QPcost = cost(res_cons.x) + Qfun_sel[i]
    else:
        feasible = 0
        QPcost = 10000
        # print("Infeasibility")
        # print( (np.dot(EqConstrCheck, EqConstrCheck) < 1e-8) )
        # print( EqConstrCheck, SS_sel[:, i] )
        # print("Term Point ", SS_sel[:, i], SS_sel, i)
        # print( ( ((b - np.dot(F,res_cons.x))).all > -1e-8), (IneqConstCheck > -1e-8).all() )
        # print(b-np.dot(F,res_cons.x))

    return res_cons.x, feasible, QPcost

def FTOCP_LMPC_CVX(M, G, E, n, SS_sel, F, b, x0, np, Qfun_sel, qp, matrix, spmatrix, i):
    if SS_sel[0,i] >=10000:
        Cost = 10000
        feasible = 0
        Solution = 10000
    else:
        ind = range(E[i].size[0] - n, E[i].size[0])
        TermPoint_sparse = spmatrix(SS_sel[:, i], ind, [0, 0])

        q = matrix(0.0, (M.size[0], 1))  # Vector associated with the linear cost, in this case zeros. NOTE: it must be dense

        res_cons = qp(M, q,  F[i], matrix(b[i]), G[i], E[i]* matrix(x0)+TermPoint_sparse)

        if res_cons['status'] == 'optimal':
            feasible = 1
            # IMPORTANT: Need to put a 2 because you have the 1/2 in front of the cost
            Cost = 2*res_cons['primal objective'] + Qfun_sel[i]
        else:
            feasible = 0
            Cost = 10000
        Solution = np.squeeze(res_cons['x'])

    return Solution, feasible, Cost

def FTOCP_LMPC_CVX_Cost(M, G, E, n, SS_sel, F, b, x0, np, Qfun_sel, qp, matrix, spmatrix, i):
    if SS_sel[0,i] >=10000:
        Cost = 10000
        feasible = 0
    else:
        ind = range(E[i].size[0] - n, E[i].size[0])
        TermPoint_sparse = spmatrix(SS_sel[:, i], ind, [0, 0])

        q = matrix(0.0, (M.size[0], 1))  # Vector associated with the linear cost, in this case zeros. NOTE: it must be dense

        res_cons = qp(M, q,  F[i], matrix(b[i]), G[i], E[i]* matrix(x0)+TermPoint_sparse)

        if res_cons['status'] == 'optimal':
            feasible = 1
            # IMPORTANT: Need to put a 2 because you have the 1/2 in front of the cost
            Cost = 2*res_cons['primal objective'] + Qfun_sel[i]
        else:
            feasible = 0
            Cost = 10000

    return Cost

def FTOCP_LMPC_CVX_Cost_Parallel(M, G, E, n, SS_sel, F, b, x0, np, Qfun_sel, qp, matrix, spmatrix, i):
    if SS_sel[0,i] >=10000:
        Cost = 10000
        feasible = 0
    else:
        ind = range(E.size[0] - n, E.size[0])
        TermPoint_sparse = spmatrix(SS_sel[:, i], ind, [0, 0])

        q = matrix(0.0, (M.size[0], 1))  # Vector associated with the linear cost, in this case zeros. NOTE: it must be dense

        res_cons = qp(M, q,  F, matrix(b), G, E* matrix(x0)+TermPoint_sparse)

        if res_cons['status'] == 'optimal':
            feasible = 1
            # IMPORTANT: Need to put a 2 because you have the 1/2 in front of the cost
            Cost = 2*res_cons['primal objective'] + Qfun_sel[i]
        else:
            feasible = 0
            Cost = 10000

    return Cost

def BuildMatEqConst_LMPC(G, E, N ,n ,d ,np, spmatrix):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

    TermCons = np.zeros((n, (N + 1) * n + N * d))
    TermCons[:, N * n:(N + 1) * n] = np.eye(n)

    G_LMPC = np.vstack((G, TermCons))
    E_LMPC = np.vstack((E, np.zeros((n, n))))

    TermPoint = np.zeros((np.shape(E_LMPC)[0]))

    G_LMPC_sparse = spmatrix(G_LMPC[np.nonzero(G_LMPC)], np.nonzero(G_LMPC)[0], np.nonzero(G_LMPC)[1], G_LMPC.shape)
    E_LMPC_sparse = spmatrix(E_LMPC[np.nonzero(E_LMPC)], np.nonzero(E_LMPC)[0], np.nonzero(E_LMPC)[1], E_LMPC.shape)

    return G_LMPC, E_LMPC, TermPoint, G_LMPC_sparse, E_LMPC_sparse

def LastIdea(SelectReg, SS_sel, N, i, CurrentRegion, F_region, b_region, np):
    UpdateSelectReg=np.zeros(N+1)
    UpdateSelectReg[0] = SelectReg[0]
    for l in range(0,N):
        if SS_sel[0, i-l] >= 10000:
            UpdateSelectReg[N-l] = SelectReg[0]
        else:
            UpdateSelectReg[N-l] = CurrentRegion(SS_sel[:, i-l], F_region, b_region, np, 0)  # Compute for point N+1

    return UpdateSelectReg
