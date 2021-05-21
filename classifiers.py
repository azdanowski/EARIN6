
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hinge_loss, log_loss
import pandas as pd
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

from sklearn.datasets import make_classification

models = {}
models['support vector'] = SVC(C=1, kernel='linear', gamma=0.1)
models['random forest'] = RandomForestClassifier(n_estimators=300)
models['logistic regression'] = LogisticRegression(max_iter=1000, solver='liblinear', random_state=0)
models['gradient boosting'] = GradientBoostingClassifier()
models['k neighbors'] = KNeighborsClassifier()

#random forest
def optimize_random_forest(X, y):
    param_grid = {'max_depth': [3, 5, 10, 20, 40],
                'min_samples_split': [2, 5, 10, 20]}
    base_estimator = RandomForestClassifier(random_state=0)
    #X, y = make_classification(n_samples=1000, random_state=0)
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2,
                            resource='n_estimators', max_resources=30).fit(X,y)
    print(sh.best_estimator_)
    return sh.best_estimator_

def optimize_logistic_regression(X,y):
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'solvers' : ['newton-cg', 'lbfgs', 'liblinear'],
                  'penalty': ['l1', 'l2'], 
                  'max_iter': [100, 200, 500, 1000]}
    base_estimator = LogisticRegression()
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2
                        , max_resources=30).fit(X,y)
    print(sh.best_estimator_)
    return sh.best_estimator_

def optimize_svc(X,y):
    param_grid = {'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
                  'C' : [0.01, 0.1, 1, 10, 100, 300],
                  'degree': [1, 2, 5],
                  'gamma' : [1e-6, 1e-4, 1e-2, 1, 10, 20, 100]
                }
    base_estimator = SVC(gamma='scale')
    sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2
                        , max_resources=20).fit(X,y)
    print(sh.best_estimator_)
    return sh.best_estimator_














def get_trained_model(name, x_train, y_train):
    if name in models:
        models[name].fit(x_train, y_train.values.ravel())
        return models[name]
    else:
        print("train: MODEL NOT FOUND")
        print(name)
        return None
    
def get_loss_function(name, x_test, y_test):
    if name in ['support vector', 'logistic regression']:
        pred_decision = models[name].decision_function(x_test)
        if name == 'support vector':
            loss = hinge_loss(y_test.values.ravel(), pred_decision)
        else:
            loss = log_loss(y_test.values.ravel(), pred_decision)
        print("Loss function:" + loss)
        return loss
    else:
        print("loss: MODEL NOT FOUND " + name)
        return None