import numpy as np
import pandas as pd
from mlsummary.clustering._clustering_functions import _clust_weight, _clustering_metrics, \
    _clust_centers, \
    _clustering_evaluation, _store_X, _scatter_clusters, _fm, _ari, _sil, _db, _ch


class MeanShiftSummary:
    def __init__(self, obj, X=None, labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.n_clusters = np.unique(obj.labels_).shape[0]
        self.variables = np.shape(obj.cluster_centers_)[1]
        self.SIL, self.DB, self.CH = _clustering_metrics(obj.labels_, X, digits)
        self.centers = _clust_centers(obj.cluster_centers_)
        self.labels = obj.labels_
        self.labels_names = np.unique(obj.labels_)
        self.cluster_size, self.cluster_weights = _clust_weight(obj.labels_)
        self.ARI, self.FM = _clustering_evaluation(obj.labels_, labels_true, digits)
        self.bandwidth = obj.bandwidth
        self.iter = obj.n_iter_
        self.X = _store_X(X, store_X)
        self.labels_true = labels_true

    def describe(self):
        print('Mean Shift algorithm')
        print('------------------')
        print('Number of clusters: {}'.format(self.n_clusters))
        print('Bandwidth: {}'.format(self.bandwidth))
        print('Number of iterations: {}'.format(self.iter))
        print('Labels name: {}'.format(self.labels_names))
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
        return 'MeanShift algorithm with {} clusters. \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())
    def __repr__(self):
        return 'MeanShift algorithm with {} clusters. \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())

    def plot(self, X = None, palette='Set2'):
        if X is None:
            X = self.X
        elif self.X is None:
            X = None

        labels = self.labels

        _scatter_clusters(_store_X(X, True), labels, palette)

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

