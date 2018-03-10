def ComputeFeasibleSolution(Time, A, B, M, G, F, E, b, x, u, n, d, N, optimize, np, InitialGuess, linalg, FTOCP, GetPred, time):
    for t in range(0, Time):
        # Solve the Finite Time Optimal Control Problem (FTOCP)
        start_time = time.clock()
        [SolutionOpt, feasible] = FTOCP(M, G, E, F, b, x[:, t], optimize, np, InitialGuess, linalg)
        InitialGuess = SolutionOpt.x
        [xPred, uPred] = GetPred(SolutionOpt, n, d, N, np)

        if feasible == 1:
            u[:, t] = uPred[0]
        else:
            u[:, t] = 10000
            print("ERROR: Optimization Problem Infeasible")
            break

        # Apply the input to the system
        x[:, t + 1] = np.dot(A[0], (x[:, t])) + np.dot(B[0], (u[:, t]))
        print "Solver Time ", time.clock() - start_time, "seconds"

    return x, u