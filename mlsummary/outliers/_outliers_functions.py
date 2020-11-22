import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score



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

def _class_pred(obj, X, y_pred, y_true, digits = 3):

    acc = None
    ce = None
    prc = None
    rcl = None
    f1 = None
    conf = None
    class_weight = None
    class_count = None



    if y_pred is not None:
        labs = pd.DataFrame(y_pred)
        class_weight = labs.value_counts(normalize=True)
        class_count = labs.value_counts(normalize=False)
        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            ce = 1-acc
            prc = precision_score(y_true, y_pred)
            rcl = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            conf = pd.DataFrame(confusion_matrix(y_true, y_pred))
    elif X is not None:
        y_pred = obj.predict(X)

        labs = pd.DataFrame(y_pred)
        class_weight = labs.value_counts(normalize=True)
        class_count = labs.value_counts(normalize=False)

        if y_true is not None:
            acc = accuracy_score(y_true, y_pred)
            ce = 1-acc
            prc = precision_score(y_true, y_pred)
            rcl = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            conf = pd.DataFrame(confusion_matrix(y_true, y_pred))

    return y_pred, y_true, round(class_weight,digits), class_count, round(acc,digits), \
           round(prc,digits), round(rcl,digits), round(f1,digits), conf, round(ce,digits)


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

def _scatter_clusters_outliers(X, labels, palette='Set2'):

    XX = X.copy()
    XX = pd.DataFrame(XX)

    if labels is None:
        sn.pairplot(XX, kind="scatter", palette=palette)
    else:
        lbs = pd.DataFrame(labels).replace({-1: 'Outlier'})
        XX['labels'] = lbs
        sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
    plt.show()

def _scatter_clusters_outliers_local(X, labels, factor, plot_factors, palette):

    XX = X.copy()
    XX = pd.DataFrame(XX)

    if plot_factors is False:
        sn.pairplot(XX, kind="scatter", palette=palette)
    else:
        lbs = pd.DataFrame(labels).replace({-1: 'Outlier'})
        XX['labels'] = lbs
        sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
        plot = sn.PairGrid(XX, hue='labels')
        plot.map(plt.scatter, s=factor)
    plt.show()