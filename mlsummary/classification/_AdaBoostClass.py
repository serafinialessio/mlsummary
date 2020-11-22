import numpy as np
from sklearn.tree import plot_tree

from mlsummary.classification._classification_functions import _class_pred, _store_X, _scatter_class, \
    _features_important, _prior, _cv_results


class AdaBoostClassSummary:
    def __init__(self, obj, X=None, X_train = None, y_pred=None, y_true=None,
                 y_train = None, y_true_train = None, store_X=False, prob_return=False, digits = 3):

        self.model = obj
        self.n_class = np.shape(obj.classes_)[0]
        self.labels = obj.classes_
        self.variables = obj.n_features_in_
        self.priors_weight, self.prior_size = _prior(y_true, digits)
        self.labels_pred, self.labels_true, self.y_pred_prob, self.class_weight, self.class_size, self.acc, \
            self.prc, self.rcl, self.f1, self.conf, self.y_train, self.y_pred_prob_train, \
            self.class_weight_train, self.class_size_train, self.acc_train, \
            self.prc_train, self.rcl_train, self.f1_train, self.conf_train, self.ce, self.ce_train, self.y_true_train = _class_pred(
                obj, X, X_train, y_pred, y_train, y_true, y_true_train, prob_return, digits)
        self.X = _store_X(X, store_X)
        self.X_train = _store_X(X_train, store_X)
        self.n_estimators = obj.n_estimators
        self.algorithm = obj.algorithm
        self.base_estimator_ = obj.base_estimator_
        self.learning_rate = obj.learning_rate
        self.estimator_weights_ = obj.estimator_weights_
        self.estimator_errors_ = obj.estimator_errors_
        self.feature_importances = _features_important(obj.feature_importances_, X)



    def describe(self):
        print('AdaBoost classifier algorithm')
        print('------------------')
        print('Base estimator : {}'.format(self.base_estimator_))
        print('Learning rate: {}'.format(self.learning_rate))
        print('Maximum number of estimators: {}'.format(self.n_estimators))
        if self.priors_weight is not None:
            print('Priors weight: \n {}'.format(self.priors_weight.to_frame().transpose().to_string(index=False)))
            print('Priors size: \n {}'.format(self.prior_size.to_frame().transpose().to_string(index=False)))

        if self.class_weight_train is not None:
            print('------')
            print('Train')
            print('------')
            print('Class weights: \n {}'.format(self.class_weight_train.to_frame().transpose().to_string(index=False)))
            print('Class size: \n {}'.format(self.class_size_train.to_frame().transpose().to_string(index=False)))

            if self.y_true_train is not None:
                print("Accuracy: {}".format(self.acc_train))
                print("Class Error: {}".format(self.ce_train))
                print("Precision Error: {}".format(self.prc_train))
                print("Recall: {}".format(self.rcl_train))
                print("F1: {}".format(self.f1_train))
                print("Confusion Matrix: \n {}".format(self.conf_train))

        if self.class_weight is not None:
            print('------')
            print('Test')
            print('------')
            print('Class weights: \n {}'.format(self.class_weight.to_frame().transpose().to_string(index=False)))
            print('Class size: \n {}'.format(self.class_size.to_frame().transpose().to_string(index=False)))

            if self.labels_true is not None:
                print("Accuracy: {}".format(self.acc))
                print("Class Error: {}".format(self.ce))
                print("Precision Error: {}".format(self.prc))
                print("Recall: {}".format(self.rcl))
                print("F1: {}".format(self.f1))
                print("Confusion Matrix: \n {}".format(self.conf))

    def __str__(self):
        return 'AdaBoost classifier with {} class \n Available attributes: \n {}'.format(self.n_class, self.__dict__.keys())
    def __repr__(self):
        return 'AdaBoost classifier with {} class \n Available attributes: \n {}'.format(self.n_class, self.__dict__.keys())


    def plot_class(self, X, y, palette = 'Set2'):

        if X is None:
            X = self.X
        elif self.X is None:
            X = None

        if y is None:
            y = self.y_pred

        _scatter_class(X = X, y = y, palette = palette)


## Search for both GridSearchCV and RandomizedSearchCV object

class AdaBoostClassSummaryCV(AdaBoostClassSummary):
    def __init__(self, obj, X=None, X_train = None, y_pred=None, y_true=None,
                 y_train = None, y_true_train = None, store_X=False, prob_return=False, digits = 3):
        self.estimator = obj.estimator
        self.best_model = obj.best_estimator_
        self.cv = obj.cv if obj.cv != None else 5
        self.parameters = obj.param_grid
        self.best_parameters = obj.best_params_
        self.best_score = round(obj.best_score_,digits)
        self.cv_results = _cv_results(obj.cv_results_, digits)

        super().__init__(obj.best_estimator_, X, X_train, y_pred, y_true,
                 y_train, y_true_train, store_X, prob_return, digits)

    def describe(self, best_model_describe = True):
        print('Cross validation AdaBoost classifier')
        print('------------------')
        print('Estimator: {}'.format(self.estimator))
        print('Best estimator: {}'.format(self.best_model))
        print('Cross validation: {}'.format(self.cv))
        print('Parameters: {}'.format(self.parameters))
        print('Best parameters: {}'.format(self.best_parameters))
        print('Best score: {}'.format(self.best_score))
        print('Results: \n {}'.format(round(self.cv_results[['mean_test_score', 'std_test_score','rank_test_score']],3)))

        if best_model_describe:
            print('------------------')
            super().describe()