import numpy as np
from warnings import warn

from MachineLearning.base_estimator import BaseEstimator
from MachineLearning.exceptions import *


class Cluster(BaseEstimator):
    """
        Cluster (Base Class: BaseEstimator)
            It is the base class for Clustering Algorithms.
    """


class KMeans(Cluster):
    """
        KMeans (Base Class: Cluster)
            It is the Unsupervised Clustering method. It creates `k` different
            clusters in the data.

        Parameters:
            k:
                -type: int
                -about: defines the number of clusters to create.

            init:
                -type: string
                -about: define the method (algorithm) for clustering
                        the data. Available options are: ['k-means++', 'random']
                -default: 'k-means++'

            max_iter:
                -type: int
                -about: defines the number of iterations to run at maximum to find 
                        `k` clusters.
                -default: 300

            n_init:
                -type: int
                -about: defines the number of times the same clustering algorithm to
                        run, to get more precise and accurate result.
                -default: 10

            distance:
                -type: string
                -about: define the metric to calculate the distance between data points.
                        Available options are: ['l1', 'l2']
                -default: 'l2'

        Input:
            X: numpy array of 2-Dimensional
    """
    def __init__(self, k, init = 'k-means++', max_iter = 300, n_init = 10, 
                 distance = 'l2'):
        self.k = k

        if init not in ['k-means++', 'random']:
            raise ValueError(f'init must be either `k-means++` or `random`. Received: {init}')
        self.init = init

        self.max_iter = max_iter
        self.n_init = n_init

        if distance not in ['l1', 'l2']:
            raise ValueError(f'distance must be either `l1` or `l2`. Received: {distance}')
        self.distance = distance
        
        
    def _calc_centroids(self, clusters, get_variance = False):
        if get_variance:
            var = 0
        centroids = np.empty((self.k, self.X.shape[1]))
        for c in range(self.k):
            h = self.X*np.where(clusters==c, 1, 0)
            cluster_points = h[np.where(h[:, 0]!=0)[0]]
            centroids[c, :] = np.mean(cluster_points, axis = 0)
            
            if get_variance:
                var += np.var(cluster_points, axis = 0)
            
        if get_variance:
            return centroids, var
        return centroids
    
    def _calc_distance(self, x1, x2, axis = -1):
        if self.distance == 'l2':
            return np.sqrt(np.sum((x1-x2)**2, axis = axis))
        elif self.distance == 'l1':
            return np.sum(np.abs(x1-x2), axis = axis)
    
    def _calc_clusters(self, points, centroids, return_distance=False):
        distances = self._calc_distance(points, centroids)
        clusters = np.argmin(distances, axis = 1).reshape(-1, 1)
        if return_distance:
            return clusters, distances
        return clusters
        
    def _initialize_centroids(self):
        if self.init == 'k-means++':
            rnd_centroids = list(np.random.choice(np.arange(self.X.shape[0]), size = 1))
            points = np.expand_dims(self.X, axis = 1)
            centroids = self.X[rnd_centroids, :].reshape(1, -1)
            
            for i in range(self.k-1):
                if i == 0:
                    dist = self._calc_distance(points, centroids)
                    centroids = np.concatenate([centroids, self.X[np.argmax(dist, axis = 0)]], axis = 0)
                    
                else:
                    cluster, dist = self._calc_clusters(points, centroids, return_distance=True)
                    centroids = np.concatenate([centroids, 
                                                points[
                                                    np.argmax(
                                                        dist[np.arange(cluster.shape[0]), cluster[:, 0]], 
                                                        axis = 0), 
                                                    :]], 
                                               axis = 0)
            return centroids
                    
        elif self.init == 'random':
            rnd_centroids = list(np.random.choice(np.arange(self.X.shape[0]), size=self.k))
            return self.X[rnd_centroids, :]
        
    def _find_clusters(self):
        centroids = self._initialize_centroids()
        points = np.expand_dims(self.X, axis = 1)
        
        prev_cluster = None
        for i in range(self.max_iter):
            clusters = self._calc_clusters(points, centroids)
            centroids, var = self._calc_centroids(clusters, True)
                
            if prev_cluster is not None:
                if (prev_cluster==clusters).all():
                    break
            prev_cluster = clusters.copy()
            
        return centroids, np.sum(var)
        
    def fit(self, X):
        super().fit(X, None)
        var = np.inf
        self.centroids = None
        for _ in range(self.n_init):
            c, v = self._find_clusters()
            if v<var:
                var = v
                self.centroids = c
            
    def predict(self, X):
        self._check_predict_input(X)
        if X.shape[1] != self.centroids.shape[1]:
            raise InvalidShape(self.centroids.shape[1], X.shape[1], True)
        distances = np.sum((np.expand_dims(X, axis = 1)-self.centroids)**2, axis = -1)
        return np.argmin(distances, axis=1).reshape(-1, 1)



class DBSCAN(Cluster):
    """
        DBSCAN (Base Class: Cluster)
            Density-based spatial clustering of applications with noise or DBSCAN
            is an unsupervised algorithm to cluster the data in some `n` clusters.
            The `n` is calculated by the algorithm itself and not defined as a parameter.

        Parameters:
            eps:
                -type: float
                -about: the maximum distance between points to consider it under the
                        same cluster (or core point).
                -default: 0.5

            min_samples:
                -type: int
                -about: the minimum number of points in a radius of `eps` to consider as 
                        a core point.
                -default: 5

            metric:
                -type: string
                -about: the metric used to calculate the distance between points.
                        Available options are: ['euclidean', 'manhattan']
                -default: 'euclidean'

        Input:
            X: numpy array of 2-Dimensional
    """

    def __init__(self, eps = 0.5, min_samples = 5, metric = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        
        if metric not in ['euclidean', 'manhattan']:
            raise ValueError(f'metric must be in [`euclidean`, `manhattan`]. Received {metric}')
        self.metric = metric
        
        self.max_clusters, self.max_spread = 1000, 1000
        
    def _calc_distance(self, x1, x2):
        diff = np.expand_dims(x1, axis = 1)-x2
        if self.metric == 'euclidean':
            return np.sqrt(np.sum(diff**2, axis = -1))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(diff), axis = -1)
        
    def fit(self, X):
        super().fit(X, None)
        distance = self._calc_distance(self.X, self.X)
        
        inside_ = np.where(distance<=self.eps, True, False)
        inside_circle = np.sum(inside_, axis = 1)
        
        core_points_idx = np.where(inside_circle>=self.min_samples, 1, 0)
        core_points = self.X[core_points_idx.astype(bool)]
        
        border_points_idx = np.where((inside_circle<self.min_samples)&(inside_circle>1), 1, 0)
        border_points = self.X[border_points_idx.astype(bool)]
        
        edge_points_idx = np.where(inside_circle==1, 1, 0)
        edge_points = self.X[edge_points_idx.astype(bool)]
        
        
        core_points_inside = {}
        for e, i in enumerate(core_points_idx):
            if i:
                core_points_inside[e]=list(np.argwhere(inside_[e])[:, 0])
                
        border_points_inside = {}
        for e, i in enumerate(border_points_idx):
            if i:
                border_points_inside[e] = list(np.argwhere(inside_[e])[:, 0])

        cluster = []        
        for l1 in range(self.max_clusters):
            all_core_pts = list(core_points_inside.keys())
            pts = [all_core_pts[0]]
            
            for l2 in range(self.max_spread):
                done = True
                for pt in pts:
                    for i in core_points_inside[pt]:
                        if (i in all_core_pts) and (i not in pts):
                            pts.append(i)
                            done = False  
                if done:
                    break
                    
            if l2 == (self.max_spread-1):
                warn('There is huge spread in the data. Increase `max_spread` from 1000 to more.')
                    
            for k, v in border_points_inside.items():
                for v_ in v:
                    if (v_ in pts)&(k not in pts):
                        pts.append(k)
                        
            for i in pts:
                if i in core_points_inside.keys():
                    core_points_inside.pop(i)
                elif i in border_points_inside.keys():
                    border_points_inside.pop(i)
                    
            cluster.append(pts)
            
            if len(core_points_inside)==0: break
              
        if l1 == (self.max_clusters-1):
            warn('More than 1000 clusters are formed. Increase `max_clusters` more than 1000 to increase clusters.')
            
                
        cluster.append(list(border_points_inside.keys())+list(np.argwhere(edge_points_idx)[:, 0]))
        edge_pts = True
        if len(cluster[-1])==0: 
            cluster.pop()
            edge_pts = False

            
        # cluster data
        self.clustered_data_ = None
        for e, clus in enumerate(cluster):
            if self.clustered_data_ is None:
                self.clustered_data_ = np.concatenate([self.X[clus, :], np.full((len(clus), 1), e)], axis = 1)
            else:
                if (e == (len(cluster)-1))&(edge_pts):
                    lbl = np.full((len(clus), 1), -1)
                else:
                    lbl = np.full((len(clus), 1), e)
                new_cluster = np.concatenate([self.X[clus, :], lbl], axis = 1)
                self.clustered_data_ = np.concatenate([self.clustered_data_, new_cluster], axis = 0)
                
        self.clusters_ = cluster
        
    def predict(self, X):
        self._check_predict_input(X)
        distance = self._calc_distance(self.clustered_data_[:, :-1], X)
        pred = []
        for idx in range(X.shape[0]):
            p = mode(self.clustered_data_[:, -1][(distance<=self.eps)[:, idx]])[0]
            if len(p)==1:
                pred += [int(p)]
            elif len(p)==0:
                pred += [-1]
        return np.array(pred).reshape(-1, 1)
    
