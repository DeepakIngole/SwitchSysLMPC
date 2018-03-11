import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem
from FTOCP import BuildMatEqConst, BuildMatCost, FTOCP, GetPred, BuildMatIneqConst
from ComputeFeasibleSolution import ComputeFeasibleSolution
from LMPCfunc import ComputeCost
from LMPC import LMPC, BuildMatCostLMPC, FTOCP_LMPC

import numpy as np
import time
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt


[A, B, Q, R, Q_LMPC, R_LMPC] = DefSystem(np)
# print("Print A in region 0", A[np.ix_([0],[0,1],[0,1])])
# print("Print A in region 0, but squeeze ", np.squeeze(A[np.ix_([0],[0,1],[0,1])]))

# Initialization
Time = 40                    # Number of simulation's steps
N = 3                       # Controller's horizon
n = 2                       # State Dimensions
d = 1                       # Input Dimensions
x_feasible = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u_feasible = np.zeros((d,Time))      # Initialize the closed loop input
x_opt      = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u_opt      = np.zeros((d,Time))      # Initialize the closed loop input

# Build Matrices for equality constraint Gz = Ex(0), where z is the optimization variavle which collects predicted state and input
[G, E] = BuildMatEqConst(A ,B ,N ,n ,d ,np)
[F, b] = BuildMatIneqConst(N, n, np, linalg)

# Build Matrices for inequality constraint Gz = Ex(0), where z is the optimization variavle which collects predicted state and input

# Build The matrices for the cost z^T M z,  where z is the optimization variavle which collects predicted state and input
P = np.array([[2.817354021023968,   2.060064829377980],
              [2.060064829377980,   3.743867101240135]]) # This is the terminal cost, it is the Lyapunov function from Matlab

M = BuildMatCost(Q, R, P, N, linalg)

# Printing statement for sanity check
print("Chekc one or several components of G")
print(G[np.ix_(np.array([1,2]),np.array([1]))])


# Main simulation loop
x_feasible[:,0] = np.array([-3.95,-0.05]) # Set initial Conditions
x_feasible[:,0] = np.array([1,1]) # Set initial Conditions
x_opt[:,0] = np.array([1,1]) # Set initial Conditions
InitialGuess = np.zeros(((N+1)*n+N*d))

[x_feasible, u_feasible] = ComputeFeasibleSolution(Time, A, B, M, G, F, E, b, x_feasible, u_feasible, n, d, N, optimize, np, InitialGuess, linalg, FTOCP, GetPred, time)

np.set_printoptions(precision=5,suppress=True)
print(x_feasible)
# plt.plot(x_feasible[0,:], x_feasible[1,:], 'ro')
# plt.axis([-4, 4, -4, 4])
# plt.show()

print("======= NOW STARTING LMPC CODE =========")
# Initialize the LMPC
Iteration = 10         #Need to define a priori the iterations as need to allocate memory
TimeLMPC = Time + 10

PointSS = 5            # Number of point per iteration to use into SS
SSit    = 1            # Number of Iterations to use into SS
SSindex = N            # First point + 1 to be in SS (i.e. pick N --> use N+1 point of the previous iteration in SS)
                       # IMPORTANT: Remember thing are indexed starting from 0

SS   = 10000*np.ones((n, TimeLMPC+1, Iteration))
Qfun = 10000*np.ones((TimeLMPC+1, Iteration))

x        = np.ones((n, TimeLMPC+1, Iteration))
u        = np.ones((d, TimeLMPC+0, Iteration))
Steps    = np.ones((Iteration))
Steps[0] = Time

# Initialize the 0-th iteration with the first feasible trajectory
x[:,0:(Steps[0]+1),0] = x_feasible
u[:,0:(Steps[0]+0),0] = u_feasible

# If more trajectories are needed in SS --> Need to store the first feasible trajectory multiple time (This to avoid size changing)
for i in range(0, SSit):
    SS[:,0:(Steps[0]+1),i]   = x[:,0:(Steps[0]+1),0]
    Qfun[0:(Steps[0]+1),i] = ComputeCost(Q_LMPC, R_LMPC, x[:,0:(Steps[0]+1),0], u[:,0:(Steps[0]+0),0], np, int(Steps[0]))

M_LMPC =  BuildMatCostLMPC(Q_LMPC, R_LMPC, N, np, linalg)
for it in range(SSit, Iteration):
    x[:, 0, it] = x[:, 0, 0]

    np.set_printoptions(precision=5, suppress=True)
    print("Learning Iteration:", it)
    start_time = time.clock()
    [x[:,:, it], u[:,:, it], Steps[it] ] = LMPC(A, B, x, u, it, SSit, np, M_LMPC, G, E, F, b, PointSS, SSindex, FTOCP_LMPC, n, d, N, SS, Qfun, linalg, optimize, InitialGuess, GetPred, time)

    # Update SS and Qfun after
    SS[:, 0:(Steps[it] + 1), it] = x[:, 0:(Steps[it] + 1), it]
    Qfun[0:(Steps[it] + 1), it] = ComputeCost(Q_LMPC, R_LMPC, x[:, 0:(Steps[it] + 1), it], u[:, 0:(Steps[it] + 0), it], np, int(Steps[it]))
    print("Terminated Learning Iteration: ", it, "Time Steps: ", Steps[it] ,"Cost Improvement: ", Qfun[0, it-1] - Qfun[0, it], "Solver time: ", time.clock() - start_time)

    if (Qfun[0, it-1] - Qfun[0, it]) < -10**(-10):
        print("ERROR: The cost is increasing, check the code",Qfun[0, it-1], Qfun[0, it], it)
        print("Iteration cost along the iterations: ", Qfun[0, 0:it+1])
        break
    elif (Qfun[0, it-1] - Qfun[0, it]) < 10**(-10):
        print("The LMPC has converged at iteration ", it, "The Optimal Cost is: ", Qfun[0, it])
        break

print("Iteration cost along the iterations: ",Qfun[0,0:it+1], "at iteration ", it)

# Now Compute Optimal Trajectory For Comparison
N_opt = 20
M_opt = BuildMatCost(Q_LMPC, R_LMPC, Q_LMPC, N_opt, linalg)
[G_opt, E_opt]  = BuildMatEqConst(A ,B ,N_opt ,n ,d ,np)
[F_opt, b_opt]  = BuildMatIneqConst(N_opt, n, np, linalg)
InitialGuess    = np.zeros((N_opt+1)*n+N_opt*d)
[res, feasible] = FTOCP(M_opt, G_opt, E_opt, F_opt, b_opt, x_opt[:, 0], optimize, np, InitialGuess, linalg)
[x_opt, u_opt]  = GetPred(res,n,d,N_opt, np)
print("Optimal Cost is: ", np.dot((res.x).T, np.dot(M_opt, res.x)) )


plt.plot(x[0,:,it], x[1,:,it], "r*", marker="*",  markersize=10)
plt.plot(x_opt[0,:], x_opt[1,:], 'bo', marker="o",  markersize=5)
plt.show()
