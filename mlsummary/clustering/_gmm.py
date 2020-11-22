import numpy as np
import pandas as pd
from mlsummary.clustering._clustering_functions import _clust_weight, _clustering_metrics, \
    _clust_centers, \
    _clustering_evaluation, _store_X, _scatter_clusters, _fm, _ari, _sil, _db, _ch, _clust_proba


class gmmSummary:
    def __init__(self, obj, X=None, labels=None, labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.components = obj.n_components
        self.variables = obj.n_features_in_
        self.covariance_type = obj.covariance_type
        self.centers = _clust_centers(obj.means_)
        self.covariance = obj.covariances_
        self.labels, self.proba, self.cluster_size = _clust_proba(obj, X, labels)
        self.labels_names = np.unique(self.labels)
        self.cluster_weights = pd.DataFrame(obj.weights_)
        self.ARI, self.FM = _clustering_evaluation(self.labels, labels_true, digits)
        self.SIL, self.DB, self.CH = _clustering_metrics(self.labels, X, digits)
        self.iter = obj.n_iter_
        self.init = obj.n_init
        self.init_type = obj.init_params
        self.X = _store_X(X, store_X)
        self.labels_true = labels_true

    def describe(self):
        print('Gaussian Mixture Models')
        print('--------------------')
        print('Number of clusters: {}'.format(self.components))
        print('Initialization: {}'.format(self.init_type))
        print('Number of initialisations: {}'.format(self.init))
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
        if self.cluster_size is not None:
            print('Clusters weights: \n {}'.format(self.cluster_size.to_frame().transpose().to_string(index=False)))
        print('Clusters weights: \n {}'.format(self.cluster_weights.to_frame().transpose().to_string(index=False)))
        # print('Cluster centers: \n {}'.format(self.centers))
        # print('Available attributes: \n {}'.format(self.__dict__.keys()))

    def __str__(self):
        return 'Gaussian Mixture Models with {} components and {} covariance \n Available attributes: \n {}'.format(self.components, self.covariance_type ,self.__dict__.keys())

    def __repr__(self):
        return 'Gaussian Mixture Models with {} components and {} covariance \n Available attributes: \n {}'.format(self.components, self.covariance_type ,self.__dict__.keys())

    def plot(self, X=None, palette='Set2'):
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

    def ch(self, X, labels, digits = 3):
        return _ch(X, labels, digits)

