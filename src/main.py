import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem
from FTOCP import BuildMatEqConst, BuildMatCost, FTOCP, GetPred, BuildMatIneqConst

import numpy as np
import time
from scipy import linalg
from scipy import optimize
import matplotlib.pyplot as plt


[A, B, Q, R] = DefSystem(np)
# print("Print A in region 0", A[np.ix_([0],[0,1],[0,1])])
# print("Print A in region 0, but squeeze ", np.squeeze(A[np.ix_([0],[0,1],[0,1])]))

# Initialization
Time = 10                    # Number of simulation's steps
N = 5                       # Controller's horizon
n = 2                       # State Dimensions
d = 1                       # Input Dimensions
x = np.zeros((n,Time+1))    # Initialize the closed loop trajectory
u = np.zeros((d,Time))      # Initialize the closed loop input

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
x[:,0] = np.array([-3.95,-0.05]) # Set initial Conditions
InitialGuess = np.zeros(((N+1)*n+N*d))

for t in range(0,Time):
    # Solve the Finite Time Optimal Control Problem (FTOCP)
    start_time = time.clock()
    [SolutionOpt, feasible] = FTOCP(M, G, E, F, b, x[:,t], optimize, np, InitialGuess, linalg)
    InitialGuess = SolutionOpt.x
    [xPred,uPred ] = GetPred(SolutionOpt, n, d, N, np)

    if feasible == 1:
        u[:, t] = uPred[0]
    else:
        u[:, t] = 10000
        print("ERROR: Optimization Problem Infeasible")
        break

    # Apply the input to the system
    x[:,t+1] = np.dot(A[0], (x[:,t])) + np.dot(B[0], (u[:,t]))
    print "Solver Time ",time.clock() - start_time, "seconds"

if feasible == 1:
    print x

    plt.plot(x[np.ix_([0],np.arange(Time+1))], x[np.ix_([1],np.arange(Time+1))], 'ro')
    plt.axis([-4, 4, -4, 4])
    plt.show()
