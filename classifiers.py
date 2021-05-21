
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hinge_loss

models = {}
models['linear support vector'] = LinearSVC()
models['random forest'] = RandomForestClassifier(n_estimators=300)
models['logistic regression'] = LogisticRegression(max_iter=1000, solver='liblinear', random_state=0)
models['gradient boosting'] = GradientBoostingClassifier()
models['k neighbors'] = KNeighborsClassifier()

def get_trained_model(name, x_train, y_train):
    if name in models:
        models[name].fit(x_train, y_train.values.ravel())
        return models[name]
    else:
        print("MODEL NOT FOUND")
        print(name)
        return None
    
def get_loss_function(name, x_test, y_test):
    if name in ['linear support vector', 'logistic regression' ]:
        pred_decision = models[name].decision_function(x_test)
        loss = hinge_loss(y_test.values.ravel(), pred_decision)
        print(loss)
        return loss
    else:
        print("MODEL NOT FOUND")
        print(name)
        return None