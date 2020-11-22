from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def _store_X(X, store_X):
    if store_X is False or X is None:
        return None
    else:
        if isinstance(X, pd.DataFrame):
            return X
        else:
            return pd.DataFrame(X)

def _prior(y_true, digits):

    class_weight = None
    class_count = None

    if y_true is not None:
        labs = pd.DataFrame(y_true)
        class_weight = round(labs.value_counts(normalize=True), digits)
        class_count = labs.value_counts(normalize=False)

    return class_weight, class_count

def _class_pred(obj, X, X_train, y_pred, y_train, y_true, y_true_train, prob_return, digits = 3):

    y_pred_prob = None
    acc = None
    ce = None
    prc = None
    rcl = None
    f1 = None
    conf = None
    class_weight = None
    class_count = None

    y_pred_prob_train = None
    acc_train = None
    ce_train = None
    prc_train = None
    rcl_train = None
    f1_train = None
    conf_train = None
    class_weight_train = None
    class_count_train = None


    if y_pred is not None:
        labs = pd.DataFrame(y_pred)
        class_weight = round(labs.value_counts(normalize=True), digits)
        class_count = labs.value_counts(normalize=False)
        if y_true is not None:
            if class_count.shape[0] > 2:
                average = 'micro'
            acc = round(accuracy_score(y_true, y_pred),digits)
            ce = round(1-acc,digits)
            prc = round(precision_score(y_true, y_pred, average=average),digits)
            rcl = round(recall_score(y_true, y_pred, average=average),digits)
            f1 = round(f1_score(y_true, y_pred, average = average),digits)
            conf = round(pd.DataFrame(confusion_matrix(y_true, y_pred)),digits)
    elif X is not None:
        y_pred = obj.predict(X)

        if prob_return is True:
            y_pred_prob = obj.predict_proba(X)

        labs = pd.DataFrame(y_pred)
        class_weight = round(labs.value_counts(normalize=True), digits)
        class_count = labs.value_counts(normalize=False)

        if y_true is not None:

            if class_count.shape[0] > 2:
                average = 'micro'
            acc = round(accuracy_score(y_true, y_pred),digits)
            ce = round(1-acc,digits)
            prc = round(precision_score(y_true, y_pred, average=average),digits)
            rcl = round(recall_score(y_true, y_pred, average=average),digits)
            f1 = round(f1_score(y_true, y_pred, average = average),digits)
            conf = round(pd.DataFrame(confusion_matrix(y_true, y_pred)),digits)



    if y_train is not None:
        labs = pd.DataFrame(y_train)
        class_weight_train = round(labs.value_counts(normalize=True),digits)
        class_count_train = labs.value_counts(normalize=False)
        if y_true_train is not None:

            if class_count_train.shape[0] > 2:
                average = 'micro'

            acc_train = round(accuracy_score(y_true_train, y_train), digits)
            ce_train = round(1-acc_train, digits)
            prc_train = round(precision_score(y_true_train, y_train, average=average), digits)
            rcl_train = round(recall_score(y_true_train, y_train, average=average), digits)
            f1_train = round(f1_score(y_true_train, y_train, average = average), digits)
            conf_train = round(pd.DataFrame(confusion_matrix(y_true_train, y_train)), digits)
    elif X_train is not None:
        y_train = obj.predict(X_train)

        if prob_return is True:
            y_pred_prob_train = obj.predict_proba(X_train)

        labs = pd.DataFrame(y_train)
        class_weight_train = round(labs.value_counts(normalize=True),digits)
        class_count_train = labs.value_counts(normalize=False)

        if y_true_train is not None:

            if class_count_train.shape[0] > 2:
                average = 'micro'

            acc_train = round(accuracy_score(y_true_train, y_train), digits)
            ce_train = round(1-acc_train, digits)
            prc_train = round(precision_score(y_true_train, y_train, average=average), digits)
            rcl_train = round(recall_score(y_true_train, y_train, average=average), digits)
            f1_train = round(f1_score(y_true_train, y_train, average=average), digits)
            conf_train = round(pd.DataFrame(confusion_matrix(y_true_train, y_train)), digits)

    return y_pred, y_true, y_pred_prob, class_weight, class_count, acc, \
           prc, rcl, f1, conf, y_train, y_pred_prob_train, \
           class_weight_train, class_count_train, acc_train, \
           prc_train, rcl_train, f1_train, conf_train, ce, ce_train, y_true_train


def _features_important(features, X):
    if X is None:
        ft = pd.Series(features).sort_values(ascending=False)
    else:
        X = pd.DataFrame(X)
        ft = pd.Series(features, index=X.columns).sort_values(ascending=False)

    return ft

def _cv_results(results, digits):
    tbl = pd.DataFrame(results, index = results.get('params'))
    return round(tbl.drop(columns=['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time']),digits)

def _scatter_class(X, y, palette):
    if X is None:
        print('X must be provided. Otherwise, X_store=True')
        return None
    else:
        XX = X.copy()
        XX = pd.DataFrame(XX)
        if y is None:
            sn.pairplot(XX, kind="scatter", palette=palette)
        else:
            XX['labels'] = y
            sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
    plt.show()

def _conf_plot(X, y_pred, y_true, palette, plot_pred):
    if X is None:
        print('X must be provided. Otherwise, X_store=True')
        return None
    else:
        XX = pd.DataFrame(X).copy()
        if y_pred is None & y_true is None:
            sn.pairplot(XX, kind="scatter",palette=palette)
        elif y_pred is not None & plot_pred is True:
            XX['labels'] = y_pred
            sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
        elif y_true is not None:
            XX['labels'] = y_true
            sn.pairplot(XX, kind="scatter", hue="labels", palette=palette)
    plt.show()

def _important_plot(features):
    sn.barplot(x=features, y=features.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.show()


