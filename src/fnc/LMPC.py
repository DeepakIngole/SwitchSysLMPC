def LMPC(A, B, x, u, it, SSit, np, M, G_LMPC, E_LMPC, TermPoint, F, b, PointSS, SSindex, FTOCP_LMPC, FTOCP_LMPC_Sol, n, d, N, SS, Qfun, linalg, optimize, InitialGuess, GetPred, time, Parallel, p, partial):
    # =============================================================================
    # ========== This functions run the it-th LMPC closed loop iteration ==========
    # =============================================================================

    # The iteration runs until the predicted trajectory reached the terminal point (i.e. in this example point closer to the origin)
    ReachedTerminalPoint = 0         # Flag to check if the iteration is completed

    # For the selected PointSS points in the latest SSit trajectories a QP is solved
    CostSingleQP  = np.zeros((SSit, PointSS))        # This vector stores the cost of each of the (PointSS * SSit) QPs
    InputSingleQP = np.zeros((SSit, PointSS, d, N))  # This vector stores the predicted input of each of the (PointSS * SSit) QPs
    FeasiSingleQP = np.zeros((SSit, PointSS))        # This vector stores the feasibility flag each of the (PointSS * SSit) QPs

    # Initilize subset of SS
    SS_sel   = 10000*np.ones((n, PointSS*SSit))       # This vector will contain PointSS points from each of the last SSit-th trajecotries in SS
    Qfun_sel = 10000*np.ones( PointSS*SSit )          # This vector will contain cost associated with the PointSS points from each of the last SSit-th trajecotries in SS
    CostQP   = 10000*np.ones(PointSS*SSit)            # This vector will contain the cost of each QP that we will solve
    SS_term  = 10000*np.ones((n,1))                   # This vector will contain the terminal point associated with the best cost from CostQP

    # Now initialize the main loop for computing the closed loop trajectory
    t = 0  # Set time = 0
    while (ReachedTerminalPoint == 0):
        # Loop over the latest SSit trajectories to create a vector of terminal constraints
        for j in range(0, SSit):
            SS_sel[:,j*PointSS + np.arange(0, PointSS)] = SS[:, SSindex + np.arange(0, PointSS), it - 1 - j]       # Store the terminal point from the (it - 1 - j)-th iteration in the vector SS_sel
            Qfun_sel[j * PointSS + np.arange(0, PointSS)] = Qfun[SSindex + np.arange(0, PointSS), it - 1 - j]      # Store cost associated with the terminal point from the (it - 1 - j)-th iteration in the vector Qfun_sel

        if Parallel == 0:
            for i in range(0, PointSS*SSit):          # Loop over the latest PointSS*SSit points
                    # Note that the time index SSindex indicates the first point that has to be considered from the (it-1-j)-th trajectory in SS
                    CostQP[i] = FTOCP_LMPC(FTOCP_LMPC_Sol, M, G_LMPC, E_LMPC, n, SSindex, it, SS_sel, TermPoint,        # Solve the Finite Time Optimal Control Problem (FTOCP)
                                      F, b, x[:, t, it], optimize, np, InitialGuess, Qfun_sel, i)

        else:

            Fun = partial(FTOCP_LMPC, FTOCP_LMPC_Sol, M, G_LMPC, E_LMPC, n, SSindex, it, SS_sel, TermPoint,         # Create the function to iterate
                          F, b, x[:, t, it], optimize, np, InitialGuess, Qfun_sel)
            index = np.arange(0, PointSS*SSit)                                                                 # Create the index vector
            Res = p.map(Fun, index)                                                                            # Run the process in parallel
            CostQP = np.asarray(Res)                                                                                # Convert the result from list to array

        CostSingleQP = CostQP.reshape(SSit, PointSS)                            # Reshape in the more natural format: (Iteration) x (Time)
        index = np.unravel_index(CostSingleQP.argmin(), CostSingleQP.shape)     # Select the indices (Iteration) x (Time) associated with the minimum cost
        j_star = index[0]
        i_star = index[1]

        SS_term[:,0] = SS[:, SSindex+i_star,it-1-j_star]                        # Pick the terminal point associated with the best cost

        # Solve the Finite Time Optimal Control Problem (FTOCP): This time get the optimal input and also the predicted trajectory
        [Sol, Feasible, Cost] = FTOCP_LMPC_Sol(M, G_LMPC, E_LMPC, n, SSindex, it, SS_term, TermPoint,
                                        F, b, x[:, t, it], optimize, np, InitialGuess, [0], 0)

        [xPred, uPred] = GetPred(Sol, n, d, N, np)          # Unpack the predicted trajectory

        # Apply the best input.
        # First check if the best input was feasible (Note that if the QP is not feasible --> cost set to 10000 --> If best problem not feasible all QP were not feasible)
        if Feasible == 1:
            u[:, t, it] = uPred[0]                 # Extract the best input
        else:
            u[:, t, it] = 10000                                                # If not feasible set input to high number
            print("ERROR: Optimization Problem Infeasible at time", t,         # Print an error message
                  SS[:, SSindex+i_star,it-1-j_star], CostSingleQP)
            ReachedTerminalPoint = 1                                           # Terminate the loop
            break

        # Apply the input to the system
        x[:, t + 1, it] = np.dot(A[0], (x[:, t, it])) + np.dot(B[0], (u[:, t, it]))
        # print "Solver Time ", time.clock() - start_time, "seconds"

        # Now check if the terminal point used as terminal constraint in the best QP is the terminal point of our task (i.e. the point closed to the origin for the (it-1-j_star)-th trajectory)
        # This is checked propagating terminal point used as terminal constraint along the SS and checking if the value of the 0-th coordinate is equal to the initialization value
        if SS[0, SSindex+i_star+1, it-1-j_star] == 10000:  # Here we check if the 0-th coordinate the propagated point equals the initilization value
            ReachedTerminalPoint = 1                       # If so, set the flag to 1: the simulation is completed

        SSindex = SSindex + i_star + 1   # Now update the time index used for picking the first point in the trajectories of SS.
                                         # This step is crucial to guarantee recursive feasibility. It is needed the
                                         # propagated point in SS will be used as terminal constraint, so that the shifted solution is feasible for the LMPC.

        # Update time index for the simulation
        t = t+1

    # After that the predicted trajectory has planned to reach the terminal point.
    # Now apply the open-loop from last prediction. This is needed to make sure that the closed-loop trajectory does not terminated further from the origin at each iteration.
    # Here we could have solved few QPs with different time horizons. This issue comes from the fact that we want to mimic an infinite horizon control problem.
    for i in range(1, N):
        u[:, t, it] = uPred[i]                              # Extract the input from the predicted ones
        x[:, t + 1, it] = np.dot(A[0], (x[:, t, it])) + np.dot(B[0], (u[:, t, it]))     # Apply the input to the system
        t = t + 1                                                                       # Update time index for the simulation

    return x[:,:,it], u[:,:,it], t



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


def FTOCP_LMPC(FTOCP_LMPC_Sol, M, G, E, n, SSindex, it, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i):
    [Sol, feasible, QPcost] = FTOCP_LMPC_Sol(M, G, E, n, SSindex, it, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i)

    return QPcost

def FTOCP_LMPC_Sol(M, G, E, n, SSindex, it, SS_sel, TermPoint, F, b, x0, optimize, np, z0, Qfun_sel, i):
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
        # print( ((np.dot(E, x0) - np.dot(G,res_cons.x)).all < 1e-8) )
        # print(np.dot(E, x0)-np.dot(G,res_cons.x))
        # print( ( ((b - np.dot(F,res_cons.x))).all > -1e-8) )
        # print(b-np.dot(F,res_cons.x))

    return res_cons.x, feasible, QPcost

def BuildMatEqConst_LMPC(G, E, N ,n ,d ,np):
    # Update the matrices for the Equality constraint in the LMPC. Now we need an extra row to constraint the terminal point to be equal to a point in SS
    # The equality constraint has now the form: G_LMPC*z = E_LMPC*x0 + TermPoint.
    # Note that the vector TermPoint is updated to constraint the predicted trajectory into a point in SS. This is done in the FTOCP_LMPC function

    TermCons = np.zeros((n, (N + 1) * n + N * d))
    TermCons[:, N * n:(N + 1) * n] = np.eye(n)

    G_LMPC = np.vstack((G, TermCons))
    E_LMPC = np.vstack((E, np.zeros((n, n))))

    TermPoint = np.zeros((np.shape(E_LMPC)[0]))

    return G_LMPC, E_LMPC, TermPoint