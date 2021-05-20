
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_linear_support_vector_classifier(x_train, y_train):
    # initialize
    lsvc = LinearSVC(random_state=0)
    # fit model, flatten y_train to 1 row
    lsvc.fit(x_train, y_train.values.ravel())
    return lsvc

def get_random_forest_classifier(x_train, y_train):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(x_train, y_train.values.ravel())
    return rfc

def get_logistic_regression(x_train, y_train):
    lr = LogisticRegression(random_state=0, solver='liblinear')
    lr.fit(x_train, y_train.values.ravel())
    return lr