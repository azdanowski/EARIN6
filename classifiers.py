
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

models = {}
models['linear support vector'] = LinearSVC()
models['random forest'] = RandomForestClassifier()
models['logistic regression'] = LogisticRegression(max_iter=1000, solver='liblinear')
models['gradient boosting'] = GradientBoostingClassifier()
models['k neigbors'] = KNeighborsClassifier()

def get_trained_model(name, x_train, y_train):
    if name in models:
        return models[name].fit(x_train, y_train.values.ravel())