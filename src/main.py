import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem, DefineRegions, PlotRegions, CurrentRegion, SysEvolution, \
    BuildMatIneqConst, GetPred, BuildMatCost, BuildMatEqConst
from LMPCfunc import ComputeCost
from LMPC import LMPC, BuildMatCostLMPC, FTOCP_LMPC, FTOCP_LMPC_Sol, BuildMatEqConst_LMPC, \
    FTOCP_LMPC_CVX, FTOCP_LMPC_CVX_Cost, FTOCP_LMPC_CVX_Cost_Parallel
import pwa_cluster as pwac

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
#from polyhedron import Vrep, Hrep
import polytope

solvers.options['show_progress'] = False      # Turn off CVX messages

# ================================================ COMMENTS ============================================================
# Example of the LMPC for picewise affine systems. In the current example there are 3 regions. The first feasible
# trajectory is compute using the LQR gain in region 0 ---> If start inside region 0 and LQR is optimal the first
# feasible trajectory will be the optimal one (try initial condition at line 48). On the other hand if you try initial
# condition that starts in region 2 (initial condition at line 46), then the LQR gain is not optimal and the LMPC
# improves the performance until it converges

# ========================================= COMMENTS ON SOLVERS ========================================================
# Scipy solver is not stable, check the branch Test_scipy_CVX_Solver for and example where it fails

# Initialization
Time = 50                            # Number of simulation's time steps for first feasible trajectory
n = 2                                # State Dimensions
d = 1                                # Input Dimensions
x_feasible = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u_feasible = np.zeros((d,Time))      # Initialize the closed loop input

# Define System Dynamics and Cost Function
[A, B, Q, R, Q_LMPC, R_LMPC, Vertex, Box_Points] = DefSystem(np)

np.random.seed(4584005)
A_true = []; B_true = []
for a in A:
    A_true.append(a + 0.2 * (np.random.uniform(size=a.shape)-1))
for b in B: 
    B_true.append(b + 0.2 * (np.random.uniform(size=b.shape)-1))

# Compute the matrices which identify the state space regions (i.e. if in x in region i --> F_region[i]*x <= b_region[i]
#F_region, b_region = DefineRegions(Vertex, Vrep, Hrep, np)
F_region = []; b_region = [];
for box in Box_Points:
    p = polytope.box2poly(box) #Polytope(vertices=Vertex)
    F_region.append(p.A); b_region.append(p.b)

# Set initial Conditions in Region 2
x_feasible[:,0] = np.array([-1, 2.5])
# Set initial Conditions in Region 0
# x_feasible[:,0] = np.array([ 1, 0])

# ================================== COMPUTE THE FIRST FEASIBLE TRAJECTORY =============================================
K = np.array([0.4221,  1.2439]) # Pick feedback gain for the first feasible trajectory
# Time loop: apply the above feedback gain
for i in range(0, Time):
    u_feasible[:,i] = -0*np.dot(K, x_feasible[:,i])
    x_feasible[:,i+1] = SysEvolution(x_feasible[:,i], u_feasible[:,i], F_region, b_region, np, CurrentRegion, A_true, B_true)

# Use data to fit linear models
zs = []; ys = []; cluster_labels = [];
for i in range(0, Time):
    cluster_labels.append(CurrentRegion(x_feasible[:,i], F_region, b_region, np, 0))
    zs.append(np.hstack([x_feasible[:,i], u_feasible[:,i]]))
    ys.append(x_feasible[:,i+1]) 

pwa_model = pwac.ClusterPWA(np.array(zs), np.array(ys), np.array(cluster_labels))
print(A_true)
print(B_true)
print(pwa_model.thetas) # can't fit B because lack of excitation



PlotRegions(Vertex, plt, np, x_feasible)

# ======================================================================================================================
# ================== Now that we have the first feasible solution we are ready for the LMPC ============================
# ======================================================================================================================1
print("====================================== STARTING LMPC CODE ===============================================")
# Setting the LMPC parameters
NumberPlots = 0          # Show plot of predicted trajectory if NumberPlots > 0
IterationPlot = 17       # If (NumberPlots == 1) and (IterationPlot =>it >= IterationPlot + NumberPlots) ---> Show plot
N = 4                    # Controller's horizon
SwLogic   = 1            # Activate transition logic based on the evolution recorded trajectory is SS
Parallel  = 0            # Set to 1 for multicore (Note if SwLogic == 1 --> Need to start several threats ---> slower then single core (can be improved))
p = Pool(4)              # Initialize the pool for multicore
Iteration = 50           # Max number of LMPC iterations (Need to define a priori the iterations as need to allocate memory)
TimeLMPC  = Time + 20    # Max number of time steps at each LMPC iteration (If this number is exceed ---> ERROR)
PointSS   = 10           # Number of point per iteration to use into SS
SSit      = 1            # Number of Iterations to use into SS
toll      = 10**(-6)     # LMPC reaches convergence whenever J^{j} - J^{j+1} <= toll (i.e. the cost is not decreasing along the iterations)

# Create the samples safe set for each region. SS_list is a list of array and each array is the sample safe set in one
# region. Analogously, Qfun_list is a list of array with the cost assocuated to each point in SS_list (i.e. for a point
# x = SS_list[region][:, time, iteration], then Qfun_list[region][:, time, iteration] is the associated cost to go.
NumRegions = len(Vertex) # Total Number of Regions
SS_list   = []
Qfun_list = []
for i in range(0, NumRegions):
    SS_list.append( 10000*np.ones((n+1, TimeLMPC+1, Iteration)) )  # For each region i initialize each SS_i
    Qfun_list.append( 10000*np.ones((TimeLMPC+1, Iteration)) )     # For each region i initialize each Qfun_i


# Variable initialization
x            = 10000*np.ones((n, TimeLMPC+1, Iteration))  # Closed loop trajectory
u            = 10000*np.ones((d, TimeLMPC+0, Iteration))  # Input associated with closed loop trajectory
Steps        = 10000*np.ones((Iteration), dtype=int)                 # This vector collests the actual time at which each iteratin is completed (Remember: it was needed to pre-allocate memory)
IndexVec     = 10000*np.ones(TimeLMPC+1)                  # This vector will be used to assign the data to the sample safe set
TotCost      = 10000*np.ones((TimeLMPC+1, Iteration))     # This vector will be used to assign the cost to the Qfunction
SelectReg    = 10000*np.ones(N+1)             # This vector collects the region to which the candidate feasible solution belongs to
InitialGuess = np.zeros(((N+1)*n+N*d))                    # Initial guess for the QP solver

# Initialize SS and Qfun
for i in range(0, SSit):
    Steps[i] = Time                             # Set the number of steps of the i-th trajectory (This is equal to time as the i-th trajectory is the first feasible solution previously computed
    x[:, 0:(Steps[i] + 1), i] = x_feasible      # Set the i-th trajectory to be the feasible solution previously computed
    u[:, 0:(Steps[i] + 0), i] = u_feasible      # Set the i-th input for the i-th trajectory to be the input from the feasible solution previously computed

    # Build SS and Q-function using previous data (3 STEPS)
    # STEP1: For each point in the previous trajectory, check to which region it belongs
    for j in range(0, Time+1):
        IndexVec[j] = CurrentRegion(x[:, j, i], F_region, b_region, np, 1)

    # STEP2: Compute the cost to go along the realized trajectory
    TotCost[0:(Steps[i]+1), i] = ComputeCost(Q_LMPC, R_LMPC, x[:,0:(Steps[i]+1),0], u[:,0:(Steps[i]+0),0], np, int(Steps[i]))

    # STEP3: Assign the realized trajectory and the cost to go to the lists SS_list and Qfun_list
    for r in range(0, NumRegions):
        if IndexVec[IndexVec==r].size:
            SS_list[r][0:n,0:IndexVec[IndexVec == r].size, i] = x[: , IndexVec==r, 0]                 # Now initialize the Sampled Safe set (SS)
            SS_list[r][n,0:IndexVec[IndexVec == r].size, i]   = np.where(IndexVec==r)[0]              # Now initialize the Sampled Safe set (SS)
            Qfun_list[r][0:IndexVec[IndexVec == r].size, i]   = TotCost[IndexVec==r, i]

    # Print Cost and time steps for first trajectories in SS
    print("Feasible Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f" % (i, Steps[i], TotCost[0, i]))

# Build the cost matrix: Given the optimization variable z, the QP will solve to minimize z^T M z,  where z is the optimization variable which collects predicted state and input
[M_LMPC, M_LMPC_sparse] =  BuildMatCostLMPC(Q_LMPC, R_LMPC, N, linalg, np, spmatrix) # Note that this will be constant throughout the regions

# ==================================== Now start the LMPC loop for the iterations ======================================
# We start from SSit, which is the number of iterations that we used from SS. Note that above we initialized the first
# SSit iterations with the first feasible trajectory. Finally, the loop terminate at Iteraions, which is the max number
# of iterations allowed
for it in range(SSit, Iteration):
    x[:, 0, it] = x[:, 0, 0] # Set the initial conditions for the it-th iteration

    # At time t=0 the candidate solution to the LMPC is x[:, 0:N, it-1] --> need to check in which regions this N steps
    # trajectory lies. This will allows us to select the regions in the LMPC
    for j in range(0, N+1):
        SelectReg[j] = CurrentRegion(x[:, j, it-1], F_region, b_region, np, 1) # For each time steps compute the region in
                                                                               # which the feasible trajectory lies

    # SSindex is the time index which corresponds to the point is the Sample Safe set which is the terminal point of the
    # candidate feasible solution.
    SSindex = N

    startTimer = datetime.datetime.now() # Start timer for LMPC iteration
    [x[:,:, it], u[:,:, it], Steps[it] ] = LMPC(A, B, x, u, it, SSit, np,  # Solve the LMPC problem at the i-th iteration
                                                M_LMPC_sparse,
                                                PointSS, SSindex, FTOCP_LMPC_CVX,
                                                n, d, N, linalg,
                                                GetPred, Parallel, p, partial,
                                                spmatrix, qp, matrix, SelectReg, BuildMatEqConst,
                                                BuildMatEqConst_LMPC, BuildMatIneqConst, F_region, b_region,
                                                CurrentRegion, SysEvolution, TotCost, plt, Vertex,
                                                Steps, NumberPlots, IterationPlot, FTOCP_LMPC_CVX_Cost_Parallel, 
                                                SwLogic, A_true=A_true, B_true=B_true)

    # LMPC iteration is completed: Stop the timer
    endTimer = datetime.datetime.now()
    deltaTimer = endTimer - startTimer

    # Build SS and Q-function using previous data (3 STEPS)
    # STEP1: For each point in the previous trajectory, check to which region it belongs
    for j in range(0, int(Steps[it])+1):
        IndexVec[j] = CurrentRegion(x[:, j, it], F_region, b_region, np, 1)

    # STEP2: Compute the cost to go along the realized trajectory
    TotCost[0:(Steps[it] + 1), it] = ComputeCost(Q_LMPC, R_LMPC, x[:, 0:(Steps[it] + 1), it],
                                                 u[:, 0:(Steps[it] + 0), it], np, int(Steps[it]))

    # STEP3: Assign the realized trajectory and the cost to go to the lists SS_list and Qfun_list
    for r in range(0, NumRegions):
        if IndexVec[IndexVec==r].size:
            SS_list[r][0:n,0:IndexVec[IndexVec == r].size,it] = x[: , IndexVec==r, it]       # Now initialize the Sampled Safe set (SS)
            SS_list[r][n,0:IndexVec[IndexVec == r].size,it]   = np.where(IndexVec==r)[0]     # Now initialize the Sampled Safe set (SS)
            Qfun_list[r][0:IndexVec[IndexVec == r].size,it]   = TotCost[IndexVec==r, it]

    # STEP4: Update models based on collected data
    # Use data to fit linear models
    for i in range(0, Steps[it]):
        cluster_labels.append(CurrentRegion(x[:,i,it], F_region, b_region, np, 0))
        zs.append(np.hstack([x[:,i, it], u[:,i, it]]))
        ys.append(x[:,i+1, it]) 
    pwa_model = pwac.ClusterPWA(np.array(zs), np.array(ys), np.array(cluster_labels))
    print(A_true)
    print(B_true)
    print(pwa_model.thetas) 

    # Print the results from the it-th iteration
    print("Learning Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f, Cost Improvement: %.7f, Iteration time: %.3fs, Avarage MIQP solver time: %.3fs"
          %(it, Steps[it], TotCost[0, it], (TotCost[0, it-1] - TotCost[0, it]), ( (deltaTimer.total_seconds() )), (((deltaTimer.total_seconds() )) / Steps[it])) )
    # Run few checks
    # if (TotCost[0, it-1] - TotCost[0, it]) < -10**(-10):  # Sanity check: Make sure that the cost is decreasing at each iteration
    #     print("ERROR: The cost is increasing, check the code",TotCost[0, it-1], TotCost[0, it], it)
    #     print("Iteration cost along the iterations: ", TotCost[0, 0:it+1])
    #     break
    #el
    if (TotCost[0, it-1] - TotCost[0, it]) < toll:      # Check if the LMPC has converged within the used-defined tollerance
        print("The LMPC has converged at iteration %d, The Optimal Cost is: %.5f" %(it, TotCost[0, it]))
        break



# Now compute in which regions the first feasible trajectory and the steady state solution lie
list_it = []
for i in range(0, int(Steps[it])+1):
    list_it.append(CurrentRegion(x[:,i,it], F_region, b_region, np, 1))

list_start = []
for i in range(0, int(Steps[0])+1):
    list_start.append(CurrentRegion(x[:,i,0], F_region, b_region, np, 1))

print("Steps in Region 0, Firs Feasible Solution: ",list_start.count(0), " Steady State: ", list_it.count(0))
print("Steps in Region 1, Firs Feasible Solution: ",list_start.count(1), " Steady State: ", list_it.count(1))
print("Steps in Region 1, Firs Feasible Solution: ",list_start.count(2), " Steady State: ", list_it.count(2))
print("Evolution First Feasible trajecotry and Steady state \n", list_start, "\n", list_it)
# print x[:,:,it]

# ======================================================================================================================
# ================================================ PLOTS ===============================================================
# ======================================================================================================================
plt.plot(np.hstack(((Vertex[0])[:, 0], np.squeeze(Vertex[0])[0, 0])),
             np.hstack(((Vertex[0])[:, 1], np.squeeze(Vertex[0])[0, 1])), "-rs")
plt.plot(np.hstack(((Vertex[1])[:, 0], np.squeeze(Vertex[1])[0, 0])),
         np.hstack(((Vertex[1])[:, 1], np.squeeze(Vertex[1])[0, 1])), "-ks")
plt.plot(np.hstack(((Vertex[2])[:, 0], np.squeeze(Vertex[2])[0, 0])),
         np.hstack(((Vertex[2])[:, 1], np.squeeze(Vertex[2])[0, 1])), "-bs")

plt.plot(x[0,0:Steps[0]+1,0], x[1,0:Steps[0]+1,0], '-ro')
for i in range(1,it):
    plt.plot(x[0, 0:Steps[i] + 1, i], x[1, 0:Steps[i] + 1, i], '-ro')

plt.plot(x[0,0:Steps[it]+1,it], x[1,0:Steps[it]+1,it], '-bo')

plt.xlim([-2.5, 2.5])
plt.ylim([-1, 4.5])

plt.show()