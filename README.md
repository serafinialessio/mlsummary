# mlsummary
mlsummary (machine learning summary ) is a Python library of tools to summarize result from machine learning algorithm (clustering, classification and outliers detection) of several Python package, as  [scikit-learn](https://scikit-learn.org/stable/), [mlxtend](http://rasbt.github.io/mlxtend/), [XGboost](https://xgboost.readthedocs.io/en/latest/python/index.html), and [LightGBM](https://lightgbm.readthedocs.io/en/latest/).



**Available algorithms classification :**

Algorithm | Library
----|---- 
Linear discriminant analysis  | scikit-learn
Quadratic discriminant analysis  | scikit-learn
Gaussian Naive Bayes  | scikit-learn
k-nearest neighbors  | scikit-learn
Nearest centroid  | scikit-learn
Radius neighbors  | scikit-learn
SoftMax regression  | mlxtend
Classification Trees  | scikit-learn
Bagging  | scikit-learn
Random Forest  | scikit-learn
AdaBoost  | scikit-learn
Gradient Boosting  | scikit-learn
XGBBoost | XGboost
LightGBM | LightGBM
Support Vector Machine | scikit-learn



**Available algorithms clustering :**

Algorithm | Library
----|----
Kmeans  | scikit-learn
Mini Batch KMeans  | scikit-learn
Gaussian mixture models  | scikit-learn
DBSCAN  | scikit-learn
OPTICS  | scikit-learn
Affinity Propagation  | scikit-learn

**Available algorithms outliers detection :**

Algorithm | Library
----|---- 
Isolation forest  | scikit-learn
Local Outlier Factor  | scikit-learn
One Class Support Vector Machine | scikit-learn

**Examples**:

- Classification [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/serafinialessio/mlsummary/main?urlpath=lab/tree/examples/classification.ipynb)
- Grid search cross validation [![Binder](https://mybinder.org/badge_logo.svg)]()
- Clustering [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/serafinialessio/mlsummary/main?urlpath=lab/tree/examples/clustering.ipynb)
- Outliers detection [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/serafinialessio/mlsummary/main?urlpath=lab/tree/examples/outliers.ipynb)



## Usage

````
python -m  pip install git+https://github.com/serafinialessio/mlsummary.git
````