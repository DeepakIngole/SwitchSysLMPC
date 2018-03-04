import sys
sys.path.append('fnc')
from UtilityFunc import DefSystem
from FTOCP import BuildMatEqConst, BuildMatCost, FTOCP, GetPred

import numpy as np
import time
from scipy import linalg
from scipy import optimize

start_time = time.clock()

# A = lil_matrix((4,4))
# B = lil_matrix([[1, 2], [3, 4]])
# print(B.toarray())
#
# a = np.array([1,2])
#
# print(A[1,1])
# print(A[a].todense())
[A, B, Q, R] = DefSystem(np)
print("Print A in region 0", A[np.ix_([0],[0,1],[0,1])])
print("Print A in region 0, but squeeze ", np.squeeze(A[np.ix_([0],[0,1],[0,1])]))

# Initialization
time = 5                    # Number of simulation's steps
N = 2                       # Controller's horizon
n = 2                       # State Dimensions
d = 1                       # Input Dimensions
x = np.zeros((n,time+1))    # Initialize the closed loop trajectory
u = np.zeros((d,time))      # Initialize the closed loop input

# Build Matrices for equality constraint Gz = Ex(0), where z is the optimization variavle which collects predicted state and input
[G, E] = BuildMatEqConst(A ,B ,N ,n ,d ,np)

# Build The matrices for the cost z^T M z,  where z is the optimization variavle which collects predicted state and input
P = np.array([[10, 0],[0, 10]]) # This is the terminal cost, it should be the Lyapunov function
M = BuildMatCost(Q, R, P, N, linalg)

# Printing statement for sanity check
print("Chekc one or several components of G")
print(G[np.ix_(np.array([1,2]),np.array([1]))])


# Main simulation loop
x[:,0] = np.array([1,1]) # Set initial Conditions
InitialGuess = np.zeros(((N+1)*n+N*d))
for t in range(0,time):
    # Solve the Finite Time Optimal Control Problem (FTOCP)
    print(InitialGuess)
    print("Const")
    SolutionOpt = FTOCP(M, G, E, x[:,t], optimize, np, InitialGuess)
    InitialGuess = SolutionOpt.x

    print("Here")
    print(np.dot(G,SolutionOpt.x))

    [xPred,uPred ] = GetPred(SolutionOpt, n, d, N, np)
    print("Solution")
    print(SolutionOpt.x)
    print("xPred")
    print(xPred)

    u[:, t] = uPred[0]
    # Apply the input to the system
    x[:,t+1] = np.dot(A[0], (x[:,t])) + np.dot(B[0], (u[:,t]))

print("ClosedLoop")
print(x)