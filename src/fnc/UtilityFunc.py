def DefSystem(np):

    # A \in \mathbb{R}^{n \dot n \dot r} where n in the dimension of the state and r is the number of regions
    # basically A[:,:,1] is the system dynamics in region 1
    A = np.array([[[1., 1.],
                   [0., 1.]],
                  [[0.4, 0.8],
                   [0. , 0.5]]])

    # B \in \mathbb{R}^{n \dot d \dot r} where n and d are the dimensions of the state and inputs and r is the number of regions
    # basically B[:,:,1] is the B matrix in region 1
    B = np.array([[[0.],
                   [1.]],
                  [[0.],
                   [1.]]])

    Q = np.eye(2)

    R = np.array(1)

    Q_LMPC = np.eye(2)

    R_LMPC = 0.001*np.array(1)

    return A, B, Q, R, Q_LMPC, R_LMPC