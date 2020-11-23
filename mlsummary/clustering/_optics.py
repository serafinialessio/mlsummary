import numpy as np
import pandas as pd
from mlsummary.clustering._clustering_functions import _clust_weight, _clustering_metrics, \
    _clustering_evaluation, _store_X, _fm, _ari, _sil, _db, _ch, _clust_centers_X
from mlsummary.outliers._outliers_functions import _scatter_clusters_outliers


class opticsSummary:
    def __init__(self, obj, X=None, labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.n_clusters = np.unique(obj.labels_).shape[0]
        self.variables = obj.n_features_in_
        self.SIL, self.DB, self.CH = _clustering_metrics(obj.labels_, X, digits)
        self.centers = _clust_centers_X(X, obj.labels_)
        self.labels = obj.labels_
        self.labels_names = obj.n_features_in_
        self.cluster_size, self.cluster_weights = _clust_weight(obj.labels_)
        self.ARI, self.FM = _clustering_evaluation(obj.labels_, labels_true, digits)
        self.eps = obj.eps
        self.max_eps = obj.max_eps
        self.min_samples = obj.min_samples
        self.min_cluster_size = obj.min_cluster_size
        self.metric = obj.metric
        self.power_mink = obj.p
        self.leaf = obj.leaf_size
        self.algorithm_type = obj.algorithm
        self.cluster_method = obj.cluster_method
        self.xi = obj.xi
        self.X = _store_X(X, store_X)
        self.labels_true = labels_true

    def describe(self):
        print('OPTICS algorithm')
        print('------------------')
        print('Number of clusters: {}'.format(self.n_clusters))
        print('Labels name: {}'.format(self.labels_names))
        print('eps: {}'.format(self.eps))
        print('Max eps: {}'.format(self.max_eps))
        print('Metric: {}'.format(self.metric))
        print('Min samples: {}'.format(self.min_samples))
        print('Min cluster size: {}'.format(self.min_cluster_size))
        print('Algorithm: {}'.format(self.algorithm_type))
        print('Leaf size: {}'.format(self.leaf))
        if self.ARI is not None:
            print('Adjusted Rand Index: {}'.format(self.ARI))
        if self.FM is not None:
            print('Fowlkes Mallows: {}'.format(self.FM))
        if self.SIL is not None:
            print('Silhouette: {}'.format(self.SIL))
        if self.DB is not None:
            print('Davies Bouldin: {}'.format(self.DB))
        if self.CH is not None:
            print('Calinski Harabasz: {}'.format(self.CH))
        print('Clusters weights: \n {}'.format(self.cluster_size.to_frame().transpose().to_string(index = False)))
        print('Clusters weights: \n {}'.format(self.cluster_weights.to_frame().transpose().to_string(index = False)))
        #print('Cluster centers: \n {}'.format(self.centers))
        #print('Available attributes: \n {}'.format(self.__dict__.keys()))

    def __str__(self):
        return 'OPTICS algorithm with {} clusters \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())
    def __repr__(self):
        return 'OPTICS algorithm with {} clusters \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())

    def plot(self, X = None, palette='Set2'):
        if X is None:
            X = self.X

        labels = self.labels

        _scatter_clusters_outliers(_store_X(X, True), labels, palette)

    def ari(self, labels, labels_true, digits = 3):
        return _ari(labels, labels_true, digits)
    def fm(self, labels, labels_true, digits = 3):
        return _fm(labels, labels_true, digits)

    def sil(self, X, labels, digits = 3):
        return _sil(X, labels, digits)
    def db(self, X, labels, digits = 3):
        return _db(X, labels, digits)
    def ch(self,X, labels, digits = 3):
        return _ch(X, labels, digits)

