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
InitialGuess = np.zeros(((N+1)*n+N*d))

[x_feasible, u_feasible] = ComputeFeasibleSolution(Time, A, B, M, G, F, E, b, x_feasible, u_feasible, n, d, N, optimize, np, InitialGuess, linalg, FTOCP, GetPred, time)

np.set_printoptions(precision=5,suppress=True)
print(x_feasible)
plt.plot(x_feasible[0,:], x_feasible[1,:], 'ro')
plt.axis([-4, 4, -4, 4])
plt.show()

print("======= NOW STARTING LMPC CODE =========")
# Initialize the LMPC
Iteration = 10 #Need to define a priori the iterations as need to allocate memory
TimeLMPC = Time + 10

PointSS = 3             # Number of point per iteration to use into SS
SSit = 1                # Number of Iterations to use into SS
SSindex = N             # First point + 1 to be in SS (i.e. pick N --> use N+1 point of the previous iteration in SS)
                        # IMPORTANT: Remember thing are indexed starting from 0

SS   = 10000*np.ones((n, TimeLMPC+1, Iteration+SSit))
Qfun = 10000*np.ones((TimeLMPC+1, Iteration+SSit))

x        = np.ones((n, TimeLMPC+1, Iteration+SSit))
u        = np.ones((d, TimeLMPC+0, Iteration+SSit))
Steps    = np.ones((Iteration+SSit))
Steps[0] = Time

# Initialize the 0-th iteration with the first feasible trajectory
x[:,0:(Steps[0]+1),0] = x_feasible
u[:,0:(Steps[0]+0),0] = u_feasible

# If more trajectories are needed in SS --> Need to store the first feasible trajectory multiple time (This to avoid size changing)
for i in range(0, SSit):
    SS[:,0:(Steps[0]+1),i]   = x[:,0:(Steps[0]+1),0]
    Qfun[0:(Steps[0]+1),i] = ComputeCost(Q_LMPC, R_LMPC, x[:,0:(Steps[0]+1),0], u[:,0:(Steps[0]+0),0], np, int(Steps[0]))

M_LMPC =  BuildMatCostLMPC(Q_LMPC, R_LMPC, N, np, linalg)
for it in range(0, Iteration):
    x[:, 0, SSit + it] = x[:, 0, 0]
    # print("Before LMPC")
    # print(Qfun)
    print("Learning Iteration:", it)
    # print("Safe Set:", SS)
    [x[:,:, SSit+it], u[:,:, SSit+it], Steps[SSit+it] ] = LMPC(A, B, x, u, it, SSit, np, M_LMPC, G, E, F, b, PointSS, SSindex, FTOCP_LMPC, n, d, N, SS, Qfun, linalg, optimize, InitialGuess, GetPred, time)
    # Update SS and Qfun after
    SS[:, 0:(Steps[SSit+it] + 1), SSit+it] = x[:, 0:(Steps[SSit+it] + 1), SSit+it]
    Qfun[0:(Steps[SSit+it] + 1), SSit+it] = ComputeCost(Q_LMPC, R_LMPC, x[:, 0:(Steps[SSit+it] + 1), SSit+it], u[:, 0:(Steps[SSit+it] + 0), SSit+it], np, int(Steps[SSit+it]))

print(Qfun[0,:])
# plt.plot(x[np.ix_([0],np.arange(Time+1))], x[np.ix_([1],np.arange(Time+1))], 'ro')
# plt.axis([-4, 4, -4, 4])
# plt.show()
