import numpy as np
import scipy.linalg as la
import rls
import cvxpy as cvx

class ClusterPWA:
    """stores clustered points and associate affine models

    Attributes:
        kernelMapCol: function returning columns of kernel map
        smoothing: parameter for smoothing objective functions
        precomputed: boolean indicated presence of precomputed kernel map
        (optional) kernelMap: precomputed full map
        (optional) KTK: kernel map conjugate transposed with itself
    """

    def __init__(self, zs, ys, initialization, init_type = None, z_cutoff=None):
        """object initialization

        Args:
            zs, ys: datapoints with which to estimate PWA function from z->y
            ignore_dims: dimensions of z to ignore for clustering
            initialization: can be one of following
                1. an integer describing the number of clusters
                2. a tuple containing the centroids and affine functions
                3. a list of cluster labels for each data point 
        """
        self.ys = ys; self.dimy = ys[0].size
        self.zs = zs; self.dimz = zs[0].size
        if z_cutoff == None:
            self.z_cutoff = self.dimz
        else:
            self.z_cutoff = z_cutoff
        self.Nd = zs.shape[0]
        self.cov_e = np.eye(self.dimy) # to do: change?
        self.update_thetas = True

        if init_type is None:
            if isinstance(initialization, int):
                init_type = 'num_clusters'
            elif len(initialization) == 2:
                init_type = 'centroids'
            else:
                init_type = 'labels'

        if init_type == 'num_clusters':
            self.Nc = initialization
            self.thetas = np.zeros( np.hstack([self.Nc, self.dimz+1, self.dimy]))
            self.centroids = np.random.uniform(size=np.hstack([self.Nc, self.z_cutoff]))
            offset = np.amin(self.zs, axis=0)
            spread = np.amax(self.zs, axis=0) - offset
            for i in range(self.z_cutoff):
                self.centroids[:,i] = spread[i]*self.centroids[:,i] + offset[i]
            self.cov_c = [np.eye(self.z_cutoff) for i in range(self.centroids.shape[0])]
            self.cluster_labels = np.zeros(self.Nd)
        elif init_type == 'centroids':
            self.centroids, self.thetas = initialization
            self.Nc = self.centroids.shape[0]
            self.cluster_labels = np.zeros(self.Nd)
            self.cov_c = [np.eye(self.z_cutoff) for i in range(self.Nc)]
        elif init_type == 'labels':
            self.cluster_labels = initialization
            self.Nc = np.unique(self.cluster_labels).size
            self.centroids, self.thetas, self.cov_c = self.get_model_from_labels()
        elif init_type == 'labels_models':
            self.cluster_labels = initialization[0]
            self.thetas = initialization[1]
            self.Nc = len(self.thetas) #np.unique(self.cluster_labels).size
            self.thetas = self.get_updated_thetas()
            self.centroids, _, self.cov_c = self.get_model_from_labels()
        elif init_type == 'labels_models_noupdate':
            self.cluster_labels = initialization[0]
            self.thetas = initialization[1]
            self.Nc = len(self.thetas) #np.unique(self.cluster_labels).size
            self.update_thetas == False
            self.centroids, _, self.cov_c = self.get_model_from_labels()
            self.fit_clusters()
            self.determine_polytopic_regions()
            #self.centroids, _, self.cov_c = self.get_model_from_labels()

    def fit_clusters(self, verbose=False):
        """iteratively fits points to clusters and affine models

        Args:
            verbose: flag for printing centroid movement at each iteration
        """
        c_error = 100
        while c_error > 1e-6:
            c_error = self.update_clusters(verbose=verbose)
            if verbose:
                print(c_error)
        print("done")

    def determine_polytopic_regions(self, verbose=False):
        ws = self.get_polytopic_regions(verbose)
        if ws[0] is not None:
            self.region_fns = np.array(ws)
        
        for i in range(len(self.zs)):
            dot_pdt = [w.transpose().dot(np.hstack([self.zs[i,0:self.z_cutoff], [1]])) for w in self.region_fns]
            self.cluster_labels[i] = np.argmax(dot_pdt)
        if self.update_thetas:
            self.centroids, self.thetas, self.cov_c = self.get_model_from_labels()
        else:
            self.centroids, _, self.cov_c = self.get_model_from_labels()

    def update_clusters(self, verbose=False):
        """updates cluster assignment, centroids, and affine models

        Returns:
            c_error: the centroid movement during the update
        """
        # Assigning each value point to best-fit cluster
        if verbose:
            print("assigning datapoints to clusters")
        for i in range(self.Nd):
            quality_of_clusters = self.cluster_quality(self.zs[i], self.ys[i])
            cluster = np.argmin(quality_of_clusters)
            self.cluster_labels[i] = cluster
            if verbose and int(self.Nd/15) == 0:
                print('processed datapoint', i)
        # Storing the old centroid values
        centroids_old = np.copy(self.centroids)
        # updating model based on new clusters
        if verbose: print("updating models")
        if self.update_thetas:
            self.centroids, self.thetas, self.cov_c = self.get_model_from_labels()
        else:
            self.centroids, _, self.cov_c = self.get_model_from_labels()        
        c_error = np.linalg.norm(self.centroids-centroids_old, ord='fro')
        return c_error

    def cluster_quality(self, z, y):
        """evaluates the quality of the fit of (z, y) to each current cluster

        Args:
            z, y: datapoint
        Returns:
            an array of model quality for each cluster
        """
        #scaling_c = [la.pinv(la.sqrtm(self.cov_c[i])) for i in range(self.Nc)]
        scaling_c = [np.eye(self.z_cutoff) for i in range(self.Nc)]
        scaling_e = la.inv(la.sqrtm(self.cov_e))
        
        # is distz the WRONG measure of locality for PWA?
        # distz = lambda idx: np.linalg.norm(scaling_c[idx].dot(z[0:self.z_cutoff]-self.centroids[idx]),2)
        def distz(idx): 
            # print(scaling_c[idx].shape, z[0:self.z_cutoff].shape, self.centroids[idx].shape)
            return np.linalg.norm(scaling_c[idx].dot(z[0:self.z_cutoff]-self.centroids[idx]),2)
        disty = lambda idx: np.linalg.norm(scaling_e.dot(y-self.thetas[idx].transpose().dot(np.hstack([z, 1]))),2)
        
        zdists = [distz(i) for i in range(self.Nc)]
        ydists = [disty(i) for i in range(self.Nc)]
        return np.array(zdists) + np.array(ydists)

    def get_model_from_labels(self):
        """ 
        Uses the cluster labels and data to return centroids and models and spatial covariances

        Returns
            centroid, affine model, and spatial covariance for each cluster
        """
        thetas = np.zeros( np.hstack([self.Nc, self.dimz+1, self.dimy]))
        centroids = np.random.uniform(size=np.hstack([self.Nc, self.z_cutoff]))
        cov_c = [np.eye(self.z_cutoff) for i in range(self.Nc)]
        for i in range(self.Nc):
            points = [self.zs[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            points_cutoff = [self.zs[j][0:self.z_cutoff] for j in range(self.Nd) if self.cluster_labels[j] == i]
            points_y = [self.ys[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            if len(points) == 0:
                # put "random" point in this cluster -- TODO more logic
                ind = int(np.round(self.Nd*np.random.rand()))
                self.cluster_labels[ind] == i
                points = [self.zs[ind]]
                points_y = [self.ys[ind]]
            else:
                cov_c[i] = np.cov(points_cutoff, rowvar=False)
                if len(cov_c[i].shape) != 2:
                    cov_c[i] = np.array([[cov_c[i]]])
            centroids[i] = np.mean(np.array(points_cutoff), axis=0)
            ls_res = np.linalg.lstsq(np.hstack([points, np.ones([len(points),1])]), points_y)
            thetas[i] = ls_res[0] # affine fit x and y
            assert len(cov_c[i].shape) == 2, cov_c[i].shape
        return centroids, thetas, cov_c

    def get_updated_thetas(self):
        """
        Uses recursive least squares to update theta based on the data points in the cluster
        """
        thetas = np.zeros( np.hstack([self.Nc, self.dimz+1, self.dimy]))
        
        for i in range(self.Nc):
            est = rls.Estimator(self.thetas[i], np.eye(self.dimz+1))

            points = [self.zs[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            points_y = [self.ys[j] for j in range(self.Nd) if self.cluster_labels[j] == i]
            for point, y in zip(points, points_y):
                est.update(np.hstack([point,[1]]), y)
            thetas[i] = est.theta
        return thetas

    def get_polytopic_regions(self, verbose=False):
        prob, ws = cvx_cluster_problem(self.zs[:,0:self.z_cutoff], self.cluster_labels)
        prob.solve(verbose=verbose)
        assert prob.status == 'optimal', "ERROR: nonoptimal polytope regions"
        return [w.value for w in ws]

def cvx_cluster_problem(zs, labels):
    s = np.unique(labels).size
    
    Ms = []
    ms = []
    ws = []
    for i in range(s):
        selected_z = zs[np.where(labels == i)]
        num_selected = selected_z.shape[0]
        M = np.hstack([selected_z,np.ones([num_selected,1])])
        Ms.append(M); ms.append(num_selected)
        ws.append(cvx.Variable(zs[0].size + 1,1))
        
    cost = 0
    constr = []
    for i in range(s):
        for j in range(s):
            if i == j: continue;
            expr = Ms[i] * (ws[j] - ws[i]) + np.ones([ms[i],1])
            cost = cost + np.ones(ms[i]) * ( cvx.pos(expr) ) / ms[i]
            
    return cvx.Problem(cvx.Minimize(cost)), ws

def getRegionMatrices(region_fns):
    F_region = []; b_region = []
    Nr = len(region_fns)
    dim = region_fns[0].size
    print(Nr, dim)
    for i in range(Nr):
        F = np.zeros([Nr-1, dim-1])
        b = np.zeros(Nr-1)
        for j in range(Nr):
            if j < i:
                F[j,:] = (region_fns[j,:-1] - region_fns[i,:-1]).T
                b[j] = region_fns[i,-1] - region_fns[j,-1]
            if j > i:
                F[j-1,:] = (region_fns[j,:-1] - region_fns[i,:-1]).T
                b[j-1] = region_fns[i,-1] - region_fns[j,-1]
        F_region.append(F); b_region.append(b)
    return F_region, b_region

def check_equivalence(region_fns, F_region, b_region, x):
    dot_pdt = [w.T.dot(np.hstack([x, [1]])) for w in region_fns]
    region_label = np.argmax(dot_pdt)
    
    matrix_label = []
    for i in range(len(F_region)):
        if np.all(F_region[i].dot(x) <= b_region[i]):
            matrix_label.append(i)
    print(region_label, matrix_label)
    

