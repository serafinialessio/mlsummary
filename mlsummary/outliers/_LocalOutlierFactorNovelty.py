import numpy as np
import pandas as pd
from mlsummary.classification._classification_functions import _class_pred, _store_X, _scatter_class, \
    _prior


class LocalOutlierFactorNoveltySummary:
    def __init__(self, obj, X=None, y_pred=None, y_true=None, store_X = False, digits = 3):

        self.model = obj
        self.n_class = np.shape(obj.classes_)[0]
        self.labels = obj.classes_
        self.variables = obj.n_features
        self.labels_pred = y_pred
        self.labels_true = y_true
        self.priors_weight, self.prior_size = _prior(y_true, digits)
        self.y_pred, self.y_true, self.y_pred_prob, self.class_weight, self.class_size, self.acc, \
            self.prc, self.rcl, self.f1, self.conf, self.ce = _class_pred(obj, X, y_pred, y_true, digits)
        self.X = _store_X(X, store_X)
        self.n_neighbors = obj.n_neighbors
        self.radius = obj.radius
        self.metric = self.effective_metric_
        self.p = obj.p
        self.fit_method = obj._fit_method
        self.leaf_size = obj.leaf_size
        self.p = obj.p
        self.contamination = obj.offset_
        self.novelty = obj.novelty
        self.negative_outlier_factor_ = pd.DataFrame(obj.negative_outlier_factor_)



    def describe(self):
        print('Local Outlier Factor novelty algorithm')
        print('------------------')
        print('Number of class: {}'.format(self.n_class))
        print('Metric: {}'.format(self.metric))
        print('Radius: {}'.format(self.radius))
        print('Power Minkowski metric: {}'.format(self.p))
        print('Contamination: {}'.format(self.contamination))
        print('Algorithm: {}'.format(self.fit_method))
        print('Novelty detection: {}'.format(self.novelty))

        if self.class_weight_train is not None:
            print('------')
            print('Train')
            print('------')
            print('Class weights: \n {}'.format(self.class_weight_train.to_frame().transpose().to_string(index=False)))
            print('Class size: \n {}'.format(self.class_size_train.to_frame().transpose().to_string(index=False)))

            if self.y_true is not None:
                print("Accuracy: {}".format(self.acc_train))
                print("Class Error: {}".format(self.ce_train))
                print("Precision Error: {}".format(self.prc_train))
                print("Recall: {}".format(self.rcl_train))
                print("F1: {}".format(self.f1_train))
                print("Confusion Matrix: {}".format(self.conf_train))

        if self.class_weight is not None:
            print('------')
            print('Test')
            print('------')
            print('Class weights: \n {}'.format(self.class_weight.to_frame().transpose().to_string(index=False)))
            print('Class size: \n {}'.format(self.class_size.to_frame().transpose().to_string(index=False)))

            if self.y_true is not None:
                print("Accuracy: {}".format(self.acc))
                print("Class Error: {}".format(self.ce))
                print("Precision Error: {}".format(self.prc))
                print("Recall: {}".format(self.rcl))
                print("F1: {}".format(self.f1))
                print("Confusion Matrix: \n {}".format(self.conf))

    def __str__(self):
        return 'Local Outlier Factor novelty \n Available attributes: \n {}'.format(self.__dict__.keys())
    def __repr__(self):
        return 'Local Outlier Factor novelty \n Available attributes: \n {}'.format(self.__dict__.keys())


    def plot_class(self, X, y, palette = 'Set2'):

        if X is None:
            X = self.X
        elif self.X is None:
            X = None

        if y is None:
            y = self.y_pred

        _scatter_class(X = X, y = y, palette = palette)
