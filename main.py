import sys
import json
from file_reader import *
from classifiers import *
from metrics import *


def max_val(d):
    """return max value and its key of dict"""
    v = list(d.values())[1:]
    acc = [val[3] for val in v]
    k = list(d.keys())[1:]
    max_v = max(acc)
    return k[acc.index(max_v)], max_v


if __name__ == '__main__':

    # preparing dataset
    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)
    test_dataset = read_test_dataset()
    # group together training and validation
    datasets = (validation_dataset, validation_labels)

    metrics_comparison = {}
    metrics_comparison["metrics"] = ["f1", "recall", "roc_auc", "accuracy"]

    # model 1: lsvc
    name = "linear support vector"
    lsvc = get_linear_support_vector_classifier(
        training_dataset, training_labels)
    metrics_comparison[name] = display_metrics_for_validation(
        lsvc, name, *datasets)
    predict_test_data(lsvc, test_dataset)

    # model 2: random forest
    name = "random forest"
    rfc = get_random_forest_classifier(training_dataset, training_labels)
    metrics_comparison[name] = display_metrics_for_validation(
        rfc, name, *datasets)
    predict_test_data(rfc, test_dataset)

    # model 3 logistic regression:
    name = 'logistic regression'
    lr = get_logistic_regression(training_dataset, training_labels)
    metrics_comparison[name] = display_metrics_for_validation(
        lr, name, *datasets)
    predict_test_data(lr, test_dataset)

    # model 4 gradient boosting:
    name = 'gradient boosting'
    gb = get_gradient_boosing_classifier(training_dataset, training_labels)
    metrics_comparison[name] = display_metrics_for_validation(
        gb, name, *datasets)
    predict_test_data(gb, test_dataset)

    # model 5 k neighbors:
    name = 'k neigbors'
    kn = get_k_neighbors_classifier(training_dataset, training_labels)
    metrics_comparison[name] = display_metrics_for_validation(
        kn, name, *datasets)
    predict_test_data(kn, test_dataset)

    # compare accuracy of all models:
    print("the most accurate model:")
    print(max_val(metrics_comparison))
    # store accuracy in file
    with open('results/metrics_table.json', 'w') as file:
        json.dump(metrics_comparison, file)

    sys.exit(0)
