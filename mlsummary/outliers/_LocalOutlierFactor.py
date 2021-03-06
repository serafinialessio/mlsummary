import numpy as np
import pandas as pd

from mlsummary.outliers._outliers_functions import _clustering_metrics, _clust_weight, \
    _clustering_evaluation, _store_X, _ari, _fm, _sil, _db, _ch, _clust_centers_X, _scatter_clusters_outliers_local


class LocalOutlierFactorSummary:
    def __init__(self, obj, X, labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.labels = obj.fit_predict(X)
        self.n_clusters = np.unique(self.labels)
        self.variables = obj.n_features_in_
        self.SIL, self.DB, self.CH = _clustering_metrics(self.labels, X, digits)
        self.centers = _clust_centers_X(X, self.labels)
        self.labels_names = np.unique(self.labels)
        self.cluster_size, self.cluster_weights = _clust_weight(self.labels)
        self.ARI, self.FM = _clustering_evaluation(self.labels, labels_true, digits)
        self.X = _store_X(X, store_X)
        self.labels_true = labels_true
        self.n_neighbors = obj.n_neighbors_
        self.radius = obj.radius
        self.metric = obj.effective_metric_
        self.p = obj.p
        self.fit_method = obj._fit_method
        self.leaf_size = obj.leaf_size
        self.contamination = obj.offset_
        self.novelty = obj.novelty
        self.negative_outlier_factor = pd.Series(obj.negative_outlier_factor_)



    def describe(self):
        print('Local Outlier Factor algorithm')
        print('------------------')
        print('Number of clusters: {}'.format(self.n_clusters))
        print('Metric: {}'.format(self.metric))
        print('Radius: {}'.format(self.radius))
        print('Power Minkowski metric: {}'.format(self.p))
        print('Contamination: {}'.format(self.contamination))
        print('Algorithm: {}'.format(self.fit_method))
        print('Novelty detection: {}'.format(self.novelty))
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

        print('Clusters weights: \n {}'.format(self.cluster_size.to_frame().transpose().to_string(index=False)))
        print('Clusters weights: \n {}'.format(self.cluster_weights.to_frame().transpose().to_string(index=False)))

    def __str__(self):
        return 'Local Outlier Factor \n Available attributes: \n {}'.format(self.__dict__.keys())
    def __repr__(self):
        return 'Local Outlier Factor \n Available attributes: \n {}'.format(self.__dict__.keys())


    def plot(self, X = 'None', palette='Set2'):

        if X is None:
            X = self.X

        labels = self.labels
        factor = self.negative_outlier_factor
        _scatter_clusters_outliers_local(_store_X(X, True), labels, factor, False, palette)

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
