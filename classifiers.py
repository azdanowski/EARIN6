
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def get_linear_support_vector_classifier(x_train, y_train):
    # initialize
    lsvc = LinearSVC()
    # fit model, flatten y_train to 1 row
    lsvc.fit(x_train, y_train.values.ravel())
    return lsvc

def get_random_forest_classifier(x_train, y_train):
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train.values.ravel())
    return rfc

def get_logistic_regression(x_train, y_train):
    lr = LogisticRegression(max_iter=10000, solver='liblinear')
    lr.fit(x_train, y_train.values.ravel())
    return lr

def get_gradient_boosing_classifier(x_train, y_train):
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train, y_train.values.ravel())
    return gbc

def get_k_neighbors_classifier(x_train, y_train):
    knc = KNeighborsClassifier()
    knc.fit(x_train, y_train.values.ravel())
    return knc
    