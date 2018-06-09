import numpy as np
import scipy.linalg as la
import rls
import cvxpy as cvx



class ClusterPWA:
    """stores clustered points and associate affine models

    Attributes:
        TODO
    """

    def __init__(self, zs, ys, num_clusters, centroids, thetas,
                 cluster_labels, cov_c, z_cutoff=None):
        """object initialization

        Args:
            zs, ys: datapoints with which to estimate PWA function from z->y
            z_cutoff: dimensions of z to ignore for clustering
            initialization: can be one of following
                1. an integer describing the number of clusters
                2. a tuple containing the centroids and affine functions
                3. a list of cluster labels for each data point 
        """
        # TODO assertions about types and shapes
        # Initializing data
        self.ys = ys; self.dimy = ys[0].size
        self.zs = zs; self.dimz = zs[0].size
        if z_cutoff == None:
            self.z_cutoff = self.dimz
        else:
            assert z_cutoff <= self.dimz, ("Cannot ignore z dimensions, \
                                            %d > %d").format(z_cutoff, self.dimz) 
            self.z_cutoff = z_cutoff
        self.Nd = zs.shape[0]
        self.cov_e = np.eye(self.dimy) # TODO: change error model?

        # Initializing clusters and models
        self.cluster_labels = cluster_labels
        self.centroids = centroids
        self.thetas = thetas
        self.Nc = num_clusters
        self.cov_c = cov_c

        self.region_fns = None


        self.update_thetas = True

    @classmethod
    def from_num_clusters(cls, zs, ys, num_clusters, z_cutoff=None):
        dimy = ys[0].size; dimz = zs[0].size
        z_lim = dimz if z_cutoff is None else z_cutoff
        # centroids are initialized to be randomly spread over the range of the data
        centroids = np.random.uniform(size=np.hstack([num_clusters, z_lim]))
        offset = np.amin(zs, axis=0)
        spread = np.amax(zs, axis=0) - offset
        for i in range(z_lim):
            centroids[:,i] = spread[i]*centroids[:,i] + offset[i]
        # covariances are initialized as identity
        cov_c = [np.eye(z_lim) for i in range(centroids.shape[0])]
        # labels are initialized to zero
        cluster_labels = np.zeros(zs.shape[0])
        # models are initialized to zero
        thetas = np.zeros( np.hstack([num_clusters, dimz+1, dimy]))
        return cls(zs, ys, num_clusters, centroids, thetas, 
                   cluster_labels, cov_c, z_cutoff)

    @classmethod
    def from_centroids_models(cls, zs, ys, centroids, thetas, z_cutoff=None):
        z_lim = zs[0].size if z_cutoff is None else z_cutoff
        cov_c = [np.eye(z_lim) for i in range(centroids.shape[0])]
        return cls(zs, ys, len(centroids), centroids, thetas, 
                   np.zeros(zs.shape[0]), cov_c, z_cutoff)

    @classmethod
    def from_labels(cls, zs, ys, cluster_labels, z_cutoff=None):
        centroids, thetas, cov_c = ClusterPWA.get_model_from_labels(zs, ys, 
                                                     cluster_labels, z_cutoff)
        return cls(zs, ys, np.unique(self.cluster_labels).size, centroids, thetas, 
                   cluster_labels, cov_c, z_cutoff)

    def add_data(self, new_zs, new_ys):
        # TODO assertions about data size
        self.zs = np.vstack([self.zs, new_zs])
        self.ys = np.vstack([self.ys, new_ys])
        self.cluster_labels = np.hstack([self.cluster_labels, 
                                          np.zeros(new_zs.shape[0])])
        self.Nd = self.zs.shape[0]

    def add_data_update(self, new_zs, new_ys, verbose=False, full_update=True):
        Nd_old = self.Nd
        self.add_data(new_zs, new_ys)
        self.update_clusters(verbose=verbose, data_start=Nd_old)
        if full_update: self.fit_clusters(verbose=verbose)

    def fit_clusters(self, data_start=0, verbose=False):
        """iteratively fits points to clusters and affine models

        Args:
            verbose: flag for printing centroid movement at each iteration
        """
        c_error = 100
        while c_error > 1e-6:
            c_error = self.update_clusters(verbose=verbose, data_start=data_start)
            if verbose: print(c_error)
        if verbose: print("done")

    def determine_polytopic_regions(self, verbose=False):
        ws = self.get_polytopic_regions(verbose)
        if ws[0] is not None:
            self.region_fns = np.array(ws)
        
        for i in range(self.Nd):
            dot_pdt = [w.T.dot(np.hstack([self.zs[i,0:self.z_cutoff], [1]])) for w in self.region_fns]
            self.cluster_labels[i] = np.argmax(dot_pdt)
        if self.update_thetas:
            self.centroids, self.thetas, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, 
                                             self.ys, self.cluster_labels, self.z_cutoff)
        else:
            self.centroids, _, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff)
        
    def get_region_matrices(self):
        return getRegionMatrices(self.region_fns)

    def get_prediction_errors(self, new_zs=None, new_ys=None):
        estimation_errors = []
        if new_zs is None:
            # compute errors on the training data
            for i in range(self.Nd):
                idx = int(self.cluster_labels[i])
                yhat = self.thetas[idx].T.dot(np.hstack([self.zs[i], 1]))
                estimation_errors.append(yhat-self.ys[i])
        else:
            # compute errors on the test data new_zs, new_ys
            for i in range(new_zs.shape[0]):
                yhat = self.get_prediction(new_zs[i])
                estimation_errors.append(yhat-new_ys[i])
        return np.array(estimation_errors)

    def update_clusters(self, data_start=0, verbose=False):
        """updates cluster assignment, centroids, and affine models

        Returns:
            c_error: the centroid movement during the update
        """
        # Assigning each value point to best-fit cluster
        if verbose:
            print("assigning datapoints to clusters")
        for i in range(data_start, self.Nd):
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
            self.centroids, self.thetas, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff)
        else:
            self.centroids, _, self.cov_c = ClusterPWA.get_model_from_labels(self.zs, self.ys, 
                                             self.cluster_labels, self.z_cutoff)        
        try:
            c_error = np.linalg.norm(self.centroids-centroids_old, ord='fro')
        except ValueError as e:
            # TODO: deal with this better
            print(e)
            self.Nc = len(self.centroids)
            c_error = 1
        return c_error

    def cluster_quality(self, z, y, no_y = False):
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
        def distz(idx): 
            return np.linalg.norm(scaling_c[idx].dot(z[0:self.z_cutoff]-self.centroids[idx]),2)
        disty = lambda idx: np.linalg.norm(scaling_e.dot(y-self.thetas[idx].T.dot(np.hstack([z, 1]))),2)
        
        zdists = [distz(i) for i in range(self.Nc)]
        if no_y: 
            return np.array(zdists)
        ydists = [disty(i) for i in range(self.Nc)]
        return np.array(zdists) + np.array(ydists)

    @staticmethod
    def get_model_from_labels(zs, ys, labels, z_cutoff=None):
        """ 
        Uses the cluster labels and data to return centroids and models and spatial covariances

        Returns
            centroid, affine model, and spatial covariance for each cluster
        """
        dimy = ys[0].size; dimz = zs[0].size
        if z_cutoff is None:
            z_cutoff = dimz
        Nc = np.unique(labels).size
        Nd = zs.shape[0]

        thetas = np.zeros( np.hstack([Nc, dimz+1, dimy]))
        centroids = np.random.uniform(size=np.hstack([Nc, z_cutoff]))
        cov_c = [np.eye(z_cutoff) for i in range(Nc)]

        # for each cluster
        for i in range(Nc):
            # gather points within the cluster
            points = [zs[j] for j in range(Nd) if labels[j] == i]
            points_cutoff = [zs[j][0:z_cutoff] for j in range(Nd) if labels[j] == i]
            points_y = [ys[j] for j in range(Nd) if labels[j] == i]
            if len(points) == 0:
                # if empty, place a random point
                # TODO more logic
                ind = int(np.round(Nd*np.random.rand()))
                labels[ind] == i
                points = [zs[ind]]
                points_cutoff = [zs[ind][0:z_cutoff]]
                points_y = [ys[ind]]
            else:
                # compute covariance
                cov_c[i] = np.cov(points_cutoff, rowvar=False)
                if len(cov_c[i].shape) != 2:
                    cov_c[i] = np.array([[cov_c[i]]])
            # compute centroids and affine fit
            centroids[i] = np.mean(np.array(points_cutoff), axis=0)
            thetas[i] = affine_fit(points, points_y) 
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
        # TODO: smart filtering of points in clusters to use fewer
        # or iterative method
        prob, ws = cvx_cluster_problem(self.zs[:,0:self.z_cutoff], self.cluster_labels)
        # TODO check solver settings, max iter, tol, etc
        prob.solve(verbose=verbose,solver=cvx.SCS)
        if prob.status != 'optimal': print("WARNING: nonoptimal polytope regions:", prob.status)
        return [w.value for w in ws]

    def get_prediction(self, z):
        if self.region_fns is not None:
            # use region functions to assign model
            dot_pdt = [w.T.dot(np.hstack([z[0:self.z_cutoff], [1]])) for w in self.region_fns]
            idx = np.argmax(dot_pdt)
            yhat = self.thetas[idx].T.dot(np.hstack([z, 1]))
        else:
            # use clustering to assign model
            quality_of_clusters = self.cluster_quality(z, None, no_y=True)
            idx = np.argmin(quality_of_clusters)
            yhat = self.thetas[idx].T.dot(np.hstack([z, 1]))
        return yhat



def affine_fit(x,y):
        # TODO use best least squares (scipy?)
        ls_res = np.linalg.lstsq(np.hstack([x, np.ones([len(x),1])]), y)
        return ls_res[0]

def cvx_cluster_problem(zs, labels):
    s = np.unique(labels).size

    
    Ms = []
    ms = []
    ws = []
    for i,label in enumerate(np.sort(np.unique(labels))):
        selected_z = zs[np.where(labels == label)]
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

def get_PWA_Models(thetas, n, p):
    As = []; Bs = []; ds = [];
    for theta in thetas:
        assert theta.shape[0] == n+p+1
        assert theta.shape[1] == n
        As.append(theta[:n, :].copy().T)
        Bs.append(theta[n:(n+p), :].copy().T)
        ds.append(theta[-1,:].copy().T)
    return As, Bs, ds

def check_equivalence(region_fns, F_region, b_region, x):
    dot_pdt = [w.T.dot(np.hstack([x, [1]])) for w in region_fns]
    region_label = np.argmax(dot_pdt)
    
    matrix_label = []
    for i in range(len(F_region)):
        if np.all(F_region[i].dot(x) <= b_region[i]):
            matrix_label.append(i)
    print(region_label, matrix_label)

def select_nc_cross_validation(nc_list, zs, ys, initialization=None, verbose=False,
                               with_polytopic_regions=False):
    # TODO test this function
    # TODO better train/test split (multiple?)
    zs_train = zs[::2]; ys_train = ys[::2]
    zs_test = zs[1::2]; ys_test = ys[1::2]
    clustering_list = []; errors = []
    for nc in nc_list: # TODO parallel?
        if verbose: print("Fitting model with Nc=", nc)
        # TODO make initialization standard
        clustering = ClusterPWA.from_num_clusters(zs_train, ys_train, nc)
        clustering_list.append(clustering)
        clustering.fit_clusters(verbose=verbose)
        if with_polytopic_regions:
            clustering.determine_polytopic_regions()
        train_errors = np.abs(clustering.get_prediction_errors(new_zs=zs2, new_ys=ys2))
        test_errors = np.abs(clustering.get_prediction_errors(new_zs=zs2, new_ys=ys2))
        # TODO: best error metric?
        metric = np.linalg.norm(test_errors, norm='fro')
        errors.append(metric)
    idx_best = np.argmin(errors)
    clustering_list[idx_best].add_data_update(zs2, ys2, verbose=verbose)
    if with_polytopic_regions:
        clustering.determine_polytopic_regions()
    return clustering_list[idx_best]
    

