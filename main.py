import sys
from file_reader import *
from classifiers import *
from metrics import *


def max_val(d):
    """ a) create a list of the dict's keys and values;
        b) return the key with the max value"""
    v = list(d.values())
    k = list(d.keys())
    max_v = max(v)
    return k[v.index(max_v)], max_v


if __name__ == '__main__':

    # preparing dataset
    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)
    test_dataset = read_test_dataset()
    # group together training and validation
    datasets = (validation_dataset, validation_labels)

    accuracy_comparizon = {}

    # model 1: lsvc
    name = "linear support vector"
    lsvc = get_linear_support_vector_classifier(
        training_dataset, training_labels)
    accuracy_comparizon[name] = display_metrics_for_validation(
        lsvc, name, *datasets)
    predict_test_data(lsvc, test_dataset)

    # model 2: random forest
    name = "random forest"
    rfc = get_random_forest_classifier(training_dataset, training_labels)
    accuracy_comparizon[name] = display_metrics_for_validation(
        rfc, name, *datasets)
    predict_test_data(rfc, test_dataset)

    # model 3 logistic regression:
    name = 'logistic regression'
    lr = get_logistic_regression(training_dataset, training_labels)
    accuracy_comparizon[name] = display_metrics_for_validation(
        lr, name, *datasets)
    predict_test_data(lr, test_dataset)
    
    # model 4 gradient boosting:
    name = 'gradient boosting'
    gb = get_gradient_boosing_classifier(training_dataset, training_labels)
    accuracy_comparizon[name] = display_metrics_for_validation(
        gb, name, *datasets)
    predict_test_data(gb, test_dataset)
    
    # model 5 k neighbors:
    name = 'k neigbors'
    kn = get_k_neighbors_classifier(training_dataset, training_labels)
    accuracy_comparizon[name] = display_metrics_for_validation(
        kn, name, *datasets)
    predict_test_data(kn, test_dataset)
    
    print("the most accurate model:")
    print(max_val(accuracy_comparizon))
    sys.exit(0)
