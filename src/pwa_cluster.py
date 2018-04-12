import numpy as np
import scipy.linalg as la

class ClusterPWA:
    """stores clustered points and associate affine models

    Attributes:
        kernelMapCol: function returning columns of kernel map
        smoothing: parameter for smoothing objective functions
        precomputed: boolean indicated presence of precomputed kernel map
        (optional) kernelMap: precomputed full map
        (optional) KTK: kernel map conjugate transposed with itself
    """

    def __init__(self, zs, ys, initialization, z_cutoff=None):
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

        if isinstance(initialization, int):
            self.Nc = initialization
            self.thetas = np.zeros( np.hstack([self.Nc, self.dimz+1, self.dimy]))
            self.centroids = np.random.uniform(size=np.hstack([self.Nc, self.z_cutoff]))
            offset = np.amin(self.zs, axis=0)
            spread = np.amax(self.zs, axis=0) - offset
            for i in range(self.z_cutoff):
                self.centroids[:,i] = spread[i]*self.centroids[:,i] + offset[i]
            self.cov_c = [np.eye(self.z_cutoff) for i in range(self.centroids.shape[0])]
            self.cluster_labels = np.zeros(self.Nd)
        elif len(initialization) == 2:
            self.centroids, self.thetas = initialization
            self.Nc = self.centroids.shape[0]
            self.cluster_labels = np.zeros(self.Nd)
            self.cov_c = [np.eye(self.z_cutoff) for i in range(self.Nc)]
        else:
            self.cluster_labels = initialization
            self.Nc = np.unqiue(self.cluster_labels).size
            self.centroids, self.thetas, self.cov_c = self.get_model_from_labels()

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
        print("updating models")
        self.centroids, self.thetas, self.cov_c = self.get_model_from_labels()
        c_error = np.linalg.norm(self.centroids-centroids_old, ord='fro')
        return c_error

    def cluster_quality(self, z, y):
        """evaluates the quality of the fit of (z, y) to each current cluster

        Args:
            z, y: datapoint
        Returns:
            an array of model quality for each cluster
        """
        scaling_c = [la.pinv(la.sqrtm(self.cov_c[i])) for i in range(self.Nc)]
        scaling_e = la.inv(la.sqrtm(self.cov_e))
        
        # is distz the WRONG measure of locality for PWA?
        distz = lambda idx: np.linalg.norm(scaling_c[idx].dot(z[0:self.z_cutoff]-self.centroids[idx]),2)
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
   
# (9,) 509 (509, 1) (7,) 509
# (9,) 693 (693, 1) (7,) 693
# (9,) 17 (17, 1) (7,) 17
# (9,) 335 (335, 1) (7,) 335
# (9,) 350 (350, 1) (7,) 350
# (9,) 138 (138, 1) (7,) 138
# (9,) 153 (153, 1) (7,) 153
# (9,) 228 (228, 1) (7,) 228
# (9,) 99 (99, 1) (7,) 99
        
