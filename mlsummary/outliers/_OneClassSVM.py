import numpy as np
import pandas as pd
from mlsummary.outliers._outliers_functions import _clustering_metrics, _clust_weight, \
    _clustering_evaluation, _store_X, _clust_centers_X, _ch, _db, _sil, _fm, _ari, \
    _scatter_clusters_outliers


class OneClassSVMSummary:
    def __init__(self, obj, X, labels_true=None, store_X=False, digits = 3):
        self.model = obj
        self.labels = obj.predict(X)
        self.n_clusters = np.unique(self.labels)
        self.variables = obj.n_features_in_
        self.SIL, self.DB, self.CH = _clustering_metrics(self.labels, X, digits)
        self.centers = _clust_centers_X(X, self.labels)
        self.labels_names = np.unique(self.labels)
        self.cluster_size, self.cluster_weights = _clust_weight(self.labels)
        self.ARI, self.FM = _clustering_evaluation(self.labels, labels_true, digits)
        self.X = _store_X(X, store_X)
        self.C = obj.C
        self.kernel = obj.kernel
        self.degree = obj.degree
        self.gamma = obj.gamma
        self.shrinking = obj.shrinking
        self.support_vectors_ = pd.DataFrame(obj.support_vectors_)
        self.n_support = pd.DataFrame(obj.n_support_)



    def describe(self):
        print('One Class support vector machine algorithm')
        print('------------------')
        print('Number of clusters: {}'.format(self.n_clusters))
        print('Kernel: {}'.format(self.kernel))
        print('Regularization parameter: {}'.format(self.C))
        if self.kernel == 'poly':
            print('Degree of the polynomial kernel: {}'.format(self.degree))
        if self.kernel != 'linear':
            print('Kernel coefficient: {}'.format(self.gamma))
        print('Shrinking: {}'.format(self.shrinking))
        print('Number of support vectors: {}'.format(self.n_support.transpose().to_string(index=False)))

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
        return 'One Class support vector machine \n Available attributes: \n {}'.format(self.__dict__.keys())
    def __repr__(self):
        return 'One Class support vector machine \n Available attributes: \n {}'.format(self.__dict__.keys())


    def plot(self, X = None, palette='Set2'):
        if X is None:
            X = self.X
        elif self.X is None:
            X = None

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
