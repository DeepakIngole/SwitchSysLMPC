import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem
from FTOCP import BuildMatEqConst, BuildMatCost, FTOCP, GetPred, BuildMatIneqConst, FTOCP_CVX
from ComputeFeasibleSolution import ComputeFeasibleSolution
from LMPCfunc import ComputeCost
from LMPC import LMPC, BuildMatCostLMPC, FTOCP_LMPC, FTOCP_LMPC_Sol, BuildMatEqConst_LMPC, FTOCP_LMPC_CVX, FTOCP_LMPC_CVX_Cost
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
import datetime
from cvxopt.solvers import qp

import numpy as np
import time
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt
from cvxopt import spmatrix, matrix,solvers

solvers.options['show_progress'] = False

[A, B, Q, R, Q_LMPC, R_LMPC] = DefSystem(np)
# print("Print A in region 0", A[np.ix_([0],[0,1],[0,1])])
# print("Print A in region 0, but squeeze ", np.squeeze(A[np.ix_([0],[0,1],[0,1])]))

# Initialization for the MPC computing the first feasible solution
CVX = 1                              # if 1 ---> Use CVX
Time = 30                            # Number of simulation's time steps
N = 3                                # Controller's horizon
n = 2                                # State Dimensions
d = 1                                # Input Dimensions
x_feasible = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u_feasible = np.zeros((d,Time))      # Initialize the closed loop input
x_opt      = np.zeros((n,Time+1))    # Initialize the optimal trajectory solved over the infinite (very long) horizon
u_opt      = np.zeros((d,Time))      # Initialize the optimal input solved over the infinite (very long) horizon

# Build Matrices for equality constraint Gz = Ex(0), where z is the optimization variable which collects predicted state and input
[G, E, G_sparse, E_sparse] = BuildMatEqConst(A ,B ,N ,n ,d ,np, spmatrix)

# Build Matrices for inequality constraint Gz = Ex(0), where z is the optimization variavle which collects predicted state and input
[F, b, F_sparse] = BuildMatIneqConst(N, n, np, linalg, spmatrix)



# Build The matrices for the cost z^T M z,  where z is the optimization variable which collects predicted state and input
P = np.array([[2.817354021023968,   2.060064829377980],
              [2.060064829377980,   3.743867101240135]]) # This is the terminal cost, it is the Lyapunov function from Matlab
[M, M_sparse] = BuildMatCost(Q, R, P, N, linalg, np, spmatrix)


# ============================================================================================
# ============ Perform simulation for computing first feasible solution ======================
# ============================================================================================
#
# Set initial conditions
# x_feasible[:,0] = np.array([-3.95,-0.05]) # Set initial Conditions
x_feasible[:,0] = np.array([1,0.9])         # Set initial Conditions
x_opt[:,0] = np.array([1,1])               # Set initial Conditions
InitialGuess = np.zeros(((N+1)*n+N*d))     # Initial guess for the QP solver

# Compute closed loop trajectory
startTimer = datetime.datetime.now()
[x_feasible, u_feasible] = ComputeFeasibleSolution(Time, A, B, M, M_sparse, G, G_sparse, F, F_sparse, E, E_sparse,
                                                   b, x_feasible, u_feasible, n, d, N, optimize, np,
                                                   InitialGuess, linalg, FTOCP, FTOCP_CVX, GetPred, time, CVX,
                                                   qp, spmatrix, matrix)
endTimer = datetime.datetime.now()
deltaTimer = endTimer - startTimer
print(deltaTimer.total_seconds())

# plt.plot(x_feasible[0,:], x_feasible[1,:], 'ro')
# plt.axis([-4, 4, -4, 4])
# plt.show()
#
# ==========================================================================================================
# ============ Now that we have the first feasible solution we are ready for the LMPC ======================
# ==========================================================================================================
print("========= STARTING LMPC CODE =========")

# Setting the LMPC parameters
CVX_LMPC  = 0            # Set to 1 for CVX
Parallel  = 1            # Set to 1 for multicore
p = Pool(4)              # Initialize the pool for multicore
Iteration = 10           # Max number of LMPC iterations (Need to define a priori the iterations as need to allocate memory)
TimeLMPC  = Time + 30    # Max number of time steps at each LMPC iteration (If this number is exceed ---> ERROR)
PointSS   = 30           # Number of point per iteration to use into SS
SSit      = 2            # Number of Iterations to use into SS
toll      = 10**(-6)     # LMPC reaches convergence whenever J^{j} - J^{j+1} <= toll (i.e. the cost is not decreasing along the iterations)
SSindex   = N            # This is the time index of the first point used in SS at time t = 0. (i.e. if SSindex = N --> use x_{N} as first terminal constraint)
                         # IMPORTANT: Remember thing are indexed starting from 0 ---> have same index as state trajectory (i.e. for i = 0 pick x_0 etc ...)

# Variable initialization
SS   = 10000*np.ones((n, TimeLMPC+1, Iteration))  # Sample safe set
Qfun = 10000*np.ones((TimeLMPC+1, Iteration))     # Terminal cost
x        = np.ones((n, TimeLMPC+1, Iteration))    # Closed loop trajectory
u        = np.ones((d, TimeLMPC+0, Iteration))    # Input associated with closed loop trajectory
Steps    = np.ones((Iteration))                   # This vector collests the actual time at which each iteratin is completed (Remember: it was needed to pre-allocate memory)


# Now given the number of iteration SSit that we want to use in the LMPC we initialize the SS and Qfun
for i in range(0, SSit):
    Steps[i] = Time                                                   # Set the number of steps of the i-th trajectory (This is equal to time as the i-th trajectory is the first feasible solution previously computed
    x[:, 0:(Steps[i] + 1), i] = x_feasible                            # Set the i-th trajectory to be the feasible solution previously computed
    u[:, 0:(Steps[i] + 0), i] = u_feasible                            # Set the i-th input for the i-th trajectory to be the input from the feasible solution previously computed
    SS[:,0:(Steps[i]+1),i]    = x[:,0:(Steps[i]+1),0]                 # Now initialize the Sampled Safe set (SS)
    Qfun[0:(Steps[i]+1),i]    = ComputeCost(Q_LMPC, R_LMPC,           # Now compute the Qfun (The terminal cost Q^j)
                                            x[:,0:(Steps[i]+1),0],
                                            u[:,0:(Steps[i]+0),0],
                                            np, int(Steps[i]))

    # Print Cost and time steps for first trajectories in SS
    print("Feasible Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f" % (i, Steps[i], Qfun[0, i]))

# Build the cost matricx: Given the optimization variable z, the QP will solve to minimize z^T M z,  where z is the optimization variable which collects predicted state and input
[M_LMPC, M_LMPC_sparse] =  BuildMatCostLMPC(Q_LMPC, R_LMPC, N, linalg, np, spmatrix) # Note that this will be constant throughout the regions

# Build the matrices for the inqeuality constraint. Need to update as the terminal point belongs to SS
[G_LMPC, E_LMPC, TermPoint, G_LMPC_sparse, E_LMPC_sparse] = BuildMatEqConst_LMPC(G, E, N ,n ,d ,np, spmatrix)

# Now start the LMPC loop for the iterations. We start from SSit, which is the number of iterations that we used from SS. Note that above we initialized the first SSit iterations
# with the first feasible trajectory. Finally, the loop terminate at Iteraions, which is the max number of iterations allowed
for it in range(SSit, Iteration):
    x[:, 0, it] = x[:, 0, 0] # Set the initial conditions for the it-th iteration

    startTimer = datetime.datetime.now()
    [x[:,:, it], u[:,:, it], Steps[it] ] = LMPC(A, B, x, u, it, SSit, np, M_LMPC, G_LMPC, E_LMPC, TermPoint,  # Solve the LMPC problem at the i-th iteration
                                                M_LMPC_sparse, G_LMPC_sparse, E_LMPC_sparse,
                                                F, F_sparse, b, PointSS, SSindex, FTOCP_LMPC,
                                                FTOCP_LMPC_Sol, FTOCP_LMPC_CVX, FTOCP_LMPC_CVX_Cost,
                                                n, d, N, SS, Qfun, linalg,
                                                optimize, InitialGuess, GetPred, time, Parallel, p, partial,
                                                CVX_LMPC, spmatrix, qp, matrix)

    # Update SS and Qfun after the it-th iteration has been completed
    SS[:, 0:(Steps[it] + 1), it] = x[:, 0:(Steps[it] + 1), it]                                      # Update SS with the it-th closed loop trajectory
    Qfun[0:(Steps[it] + 1), it]  = ComputeCost(Q_LMPC, R_LMPC, x[:, 0:(Steps[it] + 1), it],         # Update the Qfun with the it-th closed loop trajectory
                                               u[:, 0:(Steps[it] + 0), it], np, int(Steps[it]))

    # Print the results from the it-th iteration
    endTimer = datetime.datetime.now()
    deltaTimer = endTimer - startTimer
    print("Learning Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f, Cost Improvement: %.7f, Iteration time: %.3fs, Avarage MIQP solver time: %.3fs"
          %(it, Steps[it], Qfun[0, it], (Qfun[0, it-1] - Qfun[0, it]), ( (deltaTimer.total_seconds() )), (((deltaTimer.total_seconds() )) / Steps[it])) )

    # Run few checks
    if (Qfun[0, it-1] - Qfun[0, it]) < -10**(-10):  # Sanity check: Make sure that the cost is decreasing at each iteration
        print("ERROR: The cost is increasing, check the code",Qfun[0, it-1], Qfun[0, it], it)
        print("Iteration cost along the iterations: ", Qfun[0, 0:it+1])
        break
    elif (Qfun[0, it-1] - Qfun[0, it]) < toll:      # Check if the LMPC has converged within the used-defined tollerance
        print("The LMPC has converged at iteration %d, The Optimal Cost is: %.5f" %(it, Qfun[0, it]))
        break

# ======================================================================================================
# ============ Compute the optimal solution over the infinite (very long) horizon ======================
# ======================================================================================================


# Now Compute Optimal Trajectory For Comparison
N_opt = 20                                                              # Pick the long horizon to mimic the inifinite horizon
[M_opt, M_opt_sparse] = BuildMatCost(Q_LMPC, R_LMPC, Q_LMPC, N_opt, linalg, np, spmatrix)             # Build the matrix for cost
[G_opt, E_opt, G_sparse, E_sparse ]  = BuildMatEqConst(A ,B ,N_opt ,n ,d ,np, spmatrix)                # Build the matrices for equality constraint
[F_opt, b_opt, F_sparse]  = BuildMatIneqConst(N_opt, n, np, linalg, spmatrix)               # Build the matrices for inequality constraint

InitialGuess    = np.zeros((N_opt+1)*n+N_opt*d)                         # Pick the initial guess
[res, feasible] = FTOCP(M_opt, G_opt, E_opt, F_opt, b_opt, x_opt[:, 0], # Solve the FTOCP for the long horizon
                        optimize, np, InitialGuess, linalg)
[x_opt, u_opt]  = GetPred(res,n,d,N_opt, np)                            # Extract the optimal solution
print("Optimal Cost from Infinite (very long) Horizon is: %.5f" %(np.dot((res).T, np.dot(M_opt, res))) )    # Finally print the cost for comparison


# Print the optimal solution and the steady state solution of the LMPC
# plt.plot(x[0,:,it], x[1,:,it], "r*", marker="*",  markersize=10)
# plt.plot(x_opt[0,:], x_opt[1,:], 'bo', marker="o",  markersize=5)
# plt.show()
