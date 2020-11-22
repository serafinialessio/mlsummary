from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def _store_X(X, store_X):
    if store_X is False:
        return None
    else:
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X, columns=X.dtype.names)


def _clust_centers(centers):
    clusters_col_names = ['Feature ' + str(x + 1) for x in range(np.shape(centers)[1])]
    ct = pd.DataFrame(centers, columns = clusters_col_names)
    return ct

def  _clust_centers_X(X, labels):
    if X is None:
        return None
    else:
        X['labels'] = labels
        return(X.groupby('labels').mean())

def _clust_n(labels):
    cluster_count = pd.DataFrame(labels).value_counts()
    return cluster_count

def _clust_weight(labels):
    labs = pd.DataFrame(labels)
    cluster_weight = labs.value_counts(normalize=True, sort = False)
    cluster_count = labs.value_counts(normalize=False, sort = False)
    return cluster_count, cluster_weight

def _clust_proba(obj, X, labels):

    proba = None
    count = None

    if labels is not None:
        count = pd.DataFrame(labels).value_counts()

    if X is not None:
        labels = obj.predict(X)
        proba = obj.predict_proba(X)
        clusters_col_names = ['Cluster ' + str(x + 1) for x in range(np.shape(proba)[1])]
        proba = pd.DataFrame(obj.predict_proba(X), columns = clusters_col_names)
        count = pd.DataFrame(labels).value_counts()

    return labels, proba, count

def _clustering_evaluation(label, labels_true, digits):
    if labels_true is None:
        FM = None
        ARI = None
    else:
        ARI = round(adjusted_rand_score(labels_true, label), digits)
        FM = round(fowlkes_mallows_score(labels_true, label),digits)

    return ARI, FM

def _clustering_metrics(labels, X, digits):
    if X is None:
        SIL = None
        DB = None
        CH = None
    else:
        SIL = round(silhouette_score(X, labels),digits)
        DB = round(davies_bouldin_score(X, labels),digits)
        CH = round(calinski_harabasz_score(X, labels),digits)

    return SIL, DB, CH



def _ari(labels, labels_true, digits):
    return round(adjusted_rand_score(labels_true, labels),digits)

def _fm(labels, labels_true,digits):
    return round(fowlkes_mallows_score(labels_true, labels),digits)

def _sil(X, labels,digits):
    return round(silhouette_score(X, labels),digits)
def _db(X, labels,digits):
    return round(davies_bouldin_score(X, labels),digits)
def _ch(X, labels,digits):
    return round(calinski_harabasz_score(X, labels),digits)

def _scatter_clusters(X, labels, palette):
    if X is None:
        return None
    else:
        XX = X.copy()
        XX = pd.DataFrame(XX)
        if labels is None:
            sn.pairplot(XX, kind="scatter", palette=palette)
        else:
            XX['labels'] = labels
            sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
    plt.show()