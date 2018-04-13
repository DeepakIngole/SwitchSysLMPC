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


# model mismatch parameters
A_ERROR = 0.1
B_ERROR = 0.0
N_ITER = 10

# Define System Dynamics and Cost Function
[A_true, B_true, Q, R, Q_LMPC, R_LMPC, Vertex, Box_Points, _] = DefSystem(np)

np.random.seed(48480085)
A = []; B = []
for a in A_true:
    mask = (a != 0)
    A.append(a + A_ERROR * (np.random.uniform(size=a.shape)-1) * mask)
for b in B_true: 
    mask = (b != 0)
    B.append(b + B_ERROR * (np.random.uniform(size=b.shape)-1) * mask)

thetas_true = []
for a,b in zip(A_true,B_true):
    thetas_true.append(np.vstack([a.T,b.T,np.zeros([1,a.shape[1]])]))
thetas_true = np.array(thetas_true)
theta_estimation = []

# Compute the matrices which identify the state space regions (i.e. if in x in region i --> F_region[i]*x <= b_region[i]
#F_region, b_region = DefineRegions(Vertex, Vrep, Hrep, np)

Vertex = []
F_region = []; b_region = [];
for box in Box_Points:
    p = polytope.box2poly(box) #Polytope(vertices=Vertex)
    Vertex.append(polytope.extreme(p))
    print(polytope.extreme(p))
    F_region.append(p.A); b_region.append(p.b)



# Use data to fit linear models
zs = []; ys = []; cluster_labels = [];


x = np.zeros([n,Time+1,N_ITER])    # Initialize the closed loop trajectory
u = np.zeros([d,Time,N_ITER])      # Initialize the closed loop input
for j in range(N_ITER):
    # Set initial Conditions in Region 2
    if j % 2:
        x[:,0,j] = np.array([-1, 2.5]) + 0.5 * np.random.uniform(size=[2]) * np.array([1,2])
    else:
        x[:,0,j] = np.array([-1, -0.5]) + 0.5 * np.random.uniform(size=[2]) * np.array([1,2])
    # Set initial Conditions in Region 0
    # x_feasible[:,0] = np.array([ 1, 0])

    K = np.array([0.4221,  1.2439])
    K = K + 0 * np.random.uniform(size=K.shape) # Pick feedback gain for the first feasible trajectory
    # Time loop: apply the above feedback gain
    for i in range(0, Time):
        u[:,i,j] = -0.1*np.dot(K, x[:,i,j]) + 0.1 * np.random.uniform()
        x[:,i+1,j] = SysEvolution(x[:,i,j], u[:,i,j], F_region, b_region, np, CurrentRegion, A_true, B_true)

    if A_ERROR > 0 or B_ERROR > 0:
        for i in range(0, Time):
            cluster_labels.append(CurrentRegion(x[:,i,j], F_region, b_region, np, 0))
            zs.append(np.hstack([x[:,i,j], u[:,i,j]]))
            ys.append(x[:,i+1,j]) 

        thetas = []
        for a,b in zip(A,B):
            thetas.append(np.vstack([a.T,b.T,np.zeros([1,a.shape[1]])]))
        thetas = np.array(thetas)
        print(np.linalg.norm(thetas-thetas_true)) # can't fit B because lack of excitation

        #pwa_model = pwac.ClusterPWA(np.array(zs), np.array(ys), np.array(cluster_labels))
        pwa_model = pwac.ClusterPWA(np.array(zs), np.array(ys), 3, z_cutoff = n)

        pwa_model.fit_clusters()
        pwa_model.determine_polytopic_regions()

        print('datapoints:', len(zs)) # can't fit B because lack of excitation
        print('estimation error:',np.linalg.norm(pwa_model.thetas-thetas_true)) # can't fit B because lack of excitation
        theta_estimation.append(np.linalg.norm(pwa_model.thetas-thetas_true))
        A_est = []; B_est = [];
        for theta in pwa_model.thetas:
            A_est.append(theta[0:n,:].T)
            B_est.append(theta[n:n+d,:].T)
        A = A_est
        B = B_est

    
    # # STEP4: Update models based on collected data
    # # Use data to fit linear models
    # if A_ERROR > 0 or B_ERROR > 0:
    #     for i in range(0, Steps[it]):
    #         if x[0,i,it] > 1000: print(i)
    #         cluster_labels.append(CurrentRegion(x[:,i,it], F_region, b_region, np, 0))
    #         zs.append(np.hstack([x[:,i, it], u[:,i, it]]))
    #         ys.append(x[:,i+1, it]) 


    #     thetas = []
    #     for a,b in zip(A,B):
    #         thetas.append(np.vstack([a.T,b.T,np.zeros([1,a.shape[1]])]))
    #     pwa_model = pwac.ClusterPWA(np.array(zs), np.array(ys), [np.array(cluster_labels), np.array(thetas)], init_type='labels_models')
    #     print("estimation error:", np.linalg.norm(pwa_model.thetas-thetas_true)) # can't fit B because lack of excitation
    #     theta_estimation.append(np.linalg.norm(pwa_model.thetas-thetas_true))
    #     A_est = []; B_est = [];
    #     for theta in pwa_model.thetas:
    #         A_est.append(theta[0:n,:].T)
    #         B_est.append(theta[n:n+d,:].T)
    #     A = A_est
    #     B = B_est


    # # Print the results from the it-th iteration
    # print("Learning Iteration: %d, Time Steps %.1f, Iteration Cost: %.5f, Cost Improvement: %.7f, Iteration time: %.3fs, Avarage MIQP solver time: %.3fs"
    #       %(it, Steps[it], TotCost[0, it], (TotCost[0, it-1] - TotCost[0, it]), ( (deltaTimer.total_seconds() )), (((deltaTimer.total_seconds() )) / Steps[it])) )
    # # Run few checks
    # # if (TotCost[0, it-1] - TotCost[0, it]) < -10**(-10):  # Sanity check: Make sure that the cost is decreasing at each iteration
    # #     print("ERROR: The cost is increasing, check the code",TotCost[0, it-1], TotCost[0, it], it)
    # #     print("Iteration cost along the iterations: ", TotCost[0, 0:it+1])
    # #     break
    # #el
    # if np.abs(TotCost[0, it-1] - TotCost[0, it]) < toll:      # Check if the LMPC has converged within the used-defined tollerance
    #     print("The LMPC has converged at iteration %d, The Optimal Cost is: %.5f" %(it, TotCost[0, it]))
    #     break



# ======================================================================================================================
# ================================================ PLOTS ===============================================================
# ======================================================================================================================

for Vertex_plot in [Vertex]:
    plt.plot(np.hstack(((Vertex_plot[0])[:, 0], np.squeeze(Vertex_plot[0])[0, 0])),
                 np.hstack(((Vertex_plot[0])[:, 1], np.squeeze(Vertex_plot[0])[0, 1])), "-rs")
    plt.plot(np.hstack(((Vertex_plot[1])[:, 0], np.squeeze(Vertex_plot[1])[0, 0])),
             np.hstack(((Vertex_plot[1])[:, 1], np.squeeze(Vertex_plot[1])[0, 1])), "-ks")
    plt.plot(np.hstack(((Vertex_plot[2])[:, 0], np.squeeze(Vertex_plot[2])[0, 0])),
             np.hstack(((Vertex_plot[2])[:, 1], np.squeeze(Vertex_plot[2])[0, 1])), "-bs")

plt.plot(x[0,0:Time+1,0], x[1,0:Time+1,0], '-ro')
for i in range(1,N_ITER):
    plt.plot(x[0, 0:Time + 1, i], x[1, 0:Time + 1, i], '-ro')

plt.xlim([-2.5, 2.5])
plt.ylim([-1, 4.5])
plt.show()

if A_ERROR > 0 or B_ERROR > 0:
    plt.show()
    print(theta_estimation)
    plt.figure(); plt.plot(theta_estimation); plt.title("model estimation error"); plt.show()
