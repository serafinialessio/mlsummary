import numpy as np
from mlsummary.clustering._clustering_functions import _clust_weight, \
    _clust_centers, \
    _clustering_evaluation, _store_X, _scatter_clusters, _fm, _ari, MAP


class fuzzykmeansSummary:
    def __init__(self, obj, X = 'None', labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.n_clusters = obj.n_clusters
        self.variables = np.shape(obj.centers)[1]
        self.centers = _clust_centers(obj.centers)
        self.labels = MAP(obj.u)
        self.cluster_size, self.cluster_weights = _clust_weight(self.labels)
        self.ARI, self.FM = _clustering_evaluation(self.labels, labels_true, digits)
        self.fuzzy = obj.m
        self.membership = obj.u
        self.X = _store_X(X, store_X)
        self.labels_true = labels_true

    def describe(self):
        print('Fuzzy K-Means algorithm')
        print('------------------')
        print('Number of clusters: {}'.format(self.n_clusters))
        print('Fuzzy parameter: {}'.format(self.fuzzy))

        if self.ARI is not None:
            print('Adjusted Rand Index: {}'.format(self.ARI))
        if self.FM is not None:
            print('Fowlkes Mallows: {}'.format(self.FM))

        print('Clusters weights: \n {}'.format(self.cluster_size.to_frame().transpose().to_string(index = False)))
        print('Clusters weights: \n {}'.format(self.cluster_weights.to_frame().transpose().to_string(index = False)))
        #print('Cluster centers: \n {}'.format(self.centers))

    def __str__(self):
        return 'kmeans algorithm with {} clusters \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())
    def __repr__(self):
        return 'kmeans algorithm with {} clusters \n Available attributes: \n {}'.format(self.n_clusters, self.__dict__.keys())

    def plot(self, X = None, palette='Set2'):
        if X is None:
            X = self.X

        labels = self.labels

        _scatter_clusters(_store_X(X, True), labels, palette)

    def ari(self, labels, labels_true, digits = 3):
        return _ari(labels, labels_true, digits)
    def fm(self, labels, labels_true, digits = 3):
        return _fm(labels, labels_true, digits)

