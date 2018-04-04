import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem, DefineRegions, PlotRegions, CurrentRegion, SysEvolution, PlotRegionsResult
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
from polyhedron import Vrep, Hrep

# Initialization for the MPC computing the first feasible solution
Time = 30                            # Number of simulation's time steps
N = 3                                # Controller's horizon
n = 2                                # State Dimensions
d = 1                                # Input Dimensions
x_feasible = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u_feasible = np.zeros((d,Time))      # Initialize the closed loop input

solvers.options['show_progress'] = False      # Turn off CVX messages

[A, B, Q, R, Q_LMPC, R_LMPC, Vertex] = DefSystem(np)

F_region, b_region = DefineRegions(Vertex, Vrep, Hrep, np)

x_feasible[:,0] = np.array([-2,0.5])          # Set initial Conditions
K = np.array([0.4221,  1.2439])

for i in range(0, Time):
    u_feasible[:,i] = -np.dot(K, x_feasible[:,i])
    x_feasible[:,i+1] = SysEvolution(x_feasible[:,i], u_feasible[:,i], F_region, b_region, np, CurrentRegion, A, B)

print(x_feasible)

PlotRegions(Vertex, plt, np, Vrep, Hrep, x_feasible)

# ==========================================================================================================
# ============ Now that we have the first feasible solution we are ready for the LMPC ======================
# ==========================================================================================================
print("========= STARTING LMPC CODE =========")

# Setting the LMPC parameters
CVX_LMPC  = 1            # Set to 1 for CVX
Parallel  = 0            # Set to 1 for multicore
p = Pool(4)              # Initialize the pool for multicore
Iteration = 50           # Max number of LMPC iterations (Need to define a priori the iterations as need to allocate memory)
TimeLMPC  = Time + 10    # Max number of time steps at each LMPC iteration (If this number is exceed ---> ERROR)
PointSS   = 10           # Number of point per iteration to use into SS
SSit      = 1            # Number of Iterations to use into SS
toll      = 10**(-6)     # LMPC reaches convergence whenever J^{j} - J^{j+1} <= toll (i.e. the cost is not decreasing along the iterations)

# Create the samples safe set for each region. SS_list is a list of array and each array is the sample safe set in one region. Analogously, Qfun_list is a list of array with the
# cost assocuated to each point in SS_list (i.e. for a point x = SS_list[region][:, time, iteration], then Qfun_list[region][:, time, iteration] is the associated cost to go
NumRegions = len(Vertex) # Total Number of Regions
SS_list   = []
Qfun_list = []
for i in range(0, NumRegions):
    SS_list.append( 10000*np.ones((n+1, TimeLMPC+1, Iteration)) )
    Qfun_list.append( 10000*np.ones((TimeLMPC+1, Iteration)) )


# Variable initialization
x            = 10000*np.ones((n, TimeLMPC+1, Iteration))  # Closed loop trajectory
u            = 10000*np.ones((d, TimeLMPC+0, Iteration))  # Input associated with closed loop trajectory
Steps        = 10000*np.ones((Iteration))                 # This vector collests the actual time at which each iteratin is completed (Remember: it was needed to pre-allocate memory)
IndexVec     = 10000*np.ones(TimeLMPC+1)                  # This vector will be used to assign the data to the sample safe set
TotCost      = 10000*np.ones((TimeLMPC+1, Iteration))     # This vector will be used to assign the cost to the Qfunction
SelectReg    = 100*np.ones(N+1).astype(int)                # This vector collects the region to which the candidate feasible solution belongs to
InitialGuess = np.zeros(((N+1)*n+N*d))                     # Initial guess for the QP solver

# Now given the number of iteration SSit that we want to use in the LMPC we initialize the SS and Qfun
for i in range(0, SSit):
    Steps[i] = Time                                                   # Set the number of steps of the i-th trajectory (This is equal to time as the i-th trajectory is the first feasible solution previously computed
    x[:, 0:(Steps[i] + 1), i] = x_feasible                            # Set the i-th trajectory to be the feasible solution previously computed
    u[:, 0:(Steps[i] + 0), i] = u_feasible                            # Set the i-th input for the i-th trajectory to be the input from the feasible solution previously computed

    # Build SS and Q-function (3 STEPS)
    # STEP1: For each point in the previous trajectory, check to which region it belongs
    for j in range(0, Time+1):
        IndexVec[j] = CurrentRegion(x[:, j, i], F_region, b_region, np)
        print x[:, j, i], IndexVec[j]

    # STEP2: Compute the cost to go along the realized trajectory
    TotCost[0:(Steps[i]+1), i] = ComputeCost(Q_LMPC, R_LMPC, x[:,0:(Steps[i]+1),0], u[:,0:(Steps[i]+0),0], np, int(Steps[i]))

    # STEP3: Assign the realized trajectory and the cost to go to the lists SS_list and Qfun_list
    for r in range(0, NumRegions):
        if IndexVec[IndexVec==r].size:
            SS_list[r][0:n,0:IndexVec[IndexVec == r].size,i] = x[: , IndexVec==r, 0]                 # Now initialize the Sampled Safe set (SS)
            SS_list[r][n,0:IndexVec[IndexVec == r].size,i]   = np.where(IndexVec==r)[0]              # Now initialize the Sampled Safe set (SS)
            Qfun_list[r][0:IndexVec[IndexVec == r].size,i]   = TotCost[IndexVec==r, i]

    # Print Cost and time steps for first trajectories in SS
    print("Feasible Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f" % (i, Steps[i], TotCost[0, i]))

# Build the cost matricx: Given the optimization variable z, the QP will solve to minimize z^T M z,  where z is the optimization variable which collects predicted state and input
[M_LMPC, M_LMPC_sparse] =  BuildMatCostLMPC(Q_LMPC, R_LMPC, N, linalg, np, spmatrix) # Note that this will be constant throughout the regions


# Now start the LMPC loop for the iterations. We start from SSit, which is the number of iterations that we used from SS. Note that above we initialized the first SSit iterations
# with the first feasible trajectory. Finally, the loop terminate at Iteraions, which is the max number of iterations allowed

print M_LMPC
for it in range(SSit, Iteration):
    x[:, 0, it] = x[:, 0, 0] # Set the initial conditions for the it-th iteration

    for j in range(0, N+1):
        SelectReg[j] = CurrentRegion(x[:, j, it-1], F_region, b_region, np)

    SSindex = int(np.where(SS_list[SelectReg[-1]][n, :, it - 1] == N)[0])

    startTimer = datetime.datetime.now()
    [x[:,:, it], u[:,:, it], Steps[it] ] = LMPC(A, B, x, u, it, SSit, np, M_LMPC,  # Solve the LMPC problem at the i-th iteration
                                                M_LMPC_sparse,
                                                PointSS, SSindex, FTOCP_LMPC,
                                                FTOCP_LMPC_Sol, FTOCP_LMPC_CVX, FTOCP_LMPC_CVX_Cost,
                                                n, d, N, SS_list, Qfun_list, linalg,
                                                optimize, InitialGuess, GetPred, time, Parallel, p, partial,
                                                CVX_LMPC, spmatrix, qp, matrix, SelectReg, BuildMatEqConst,
                                                BuildMatEqConst_LMPC, BuildMatIneqConst, F_region, b_region, CurrentRegion, SysEvolution)


    for j in range(0, int(Steps[it])+1):
        IndexVec[j] = CurrentRegion(x[:, j, it], F_region, b_region, np)

    TotCost[0:(Steps[it] + 1), it] = ComputeCost(Q_LMPC, R_LMPC, x[:, 0:(Steps[it] + 1), it],
                                                 u[:, 0:(Steps[it] + 0), it], np, int(Steps[it]))[0]

    for r in range(0, NumRegions):
        if IndexVec[IndexVec==r].size:
            SS_list[r][0:n,0:IndexVec[IndexVec == r].size,it] = x[: , IndexVec==r, it]                 # Now initialize the Sampled Safe set (SS)
            SS_list[r][n,0:IndexVec[IndexVec == r].size,it]   = np.where(IndexVec==r)[0]              # Now initialize the Sampled Safe set (SS)
            Qfun_list[r][0:IndexVec[IndexVec == r].size,it]   = TotCost[IndexVec==r, it]

    # Print the results from the it-th iteration
    endTimer = datetime.datetime.now()
    deltaTimer = endTimer - startTimer
    print("Learning Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f, Cost Improvement: %.7f, Iteration time: %.3fs, Avarage MIQP solver time: %.3fs"
          %(it, Steps[it], TotCost[0, it], (TotCost[0, it-1] - TotCost[0, it]), ( (deltaTimer.total_seconds() )), (((deltaTimer.total_seconds() )) / Steps[it])) )
    # Run few checks
    if (TotCost[0, it-1] - TotCost[0, it]) < -10**(-10):  # Sanity check: Make sure that the cost is decreasing at each iteration
        print("ERROR: The cost is increasing, check the code",TotCost[0, it-1], TotCost[0, it], it)
        print("Iteration cost along the iterations: ", TotCost[0, 0:it+1])
        break
    elif (TotCost[0, it-1] - TotCost[0, it]) < toll:      # Check if the LMPC has converged within the used-defined tollerance
        print("The LMPC has converged at iteration %d, The Optimal Cost is: %.5f" %(it, TotCost[0, it]))
        break



# print x[:,:,it]

list_it = []
for i in range(0, int(Steps[it])+1):
    list_it.append(CurrentRegion(x[:,i,it], F_region, b_region, np))

list_start = []
for i in range(0, int(Steps[0])+1):
    list_start.append(CurrentRegion(x[:,i,0], F_region, b_region, np))

print list_start.count(1), list_it.count(1)

PlotRegionsResult(Vertex, plt, np, Vrep, Hrep, x_feasible, x[:,0:(Steps[it] + 1),it])
