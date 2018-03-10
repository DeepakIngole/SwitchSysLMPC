def ComputeCost(Q, R, x, u, np, time):
    Cost = 10000 * np.ones((time+1)) # The cost has the same elements of the vector x --> time +1

    # Now compute the cost moving backwards in a Dynamic Programming (DP) fashion.
    # We start from the last element of the vector x and we sum the running cost
    for i in range(0, time+1):
        if (i == 0): # Note that for i = 0 --> pick the latest element of the vector x
            Cost[time-i] = np.dot(x[:,time-i].T, np.dot(Q, x[:,time-i]) )
        else:
            Cost[time-i] = Cost[time-i+1] + np.dot(x[:,time-i].T, np.dot(Q, x[:,time-i]) )

    return Cost