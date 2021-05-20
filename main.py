import sys
from file_reader import * #
from classifiers import *
from metrics import *

    

if __name__ == '__main__':
    
    # preparing dataset
    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)
    test_dataset = read_test_dataset()
    # group together training and validation
    datasets = (validation_dataset, validation_labels)

    
    # model 1: lsvc
    name = "linear support vector"
    lsvc = get_linear_support_vector_classifier(training_dataset, training_labels)
    display_metrics_for_validation(lsvc, name, *datasets)
    predict_test_data(lsvc, test_dataset)



    # model 2: random forest
    name = "random forest"
    rfc = get_random_forest_classifier(training_dataset, training_labels)
    display_metrics_for_validation(rfc, name, *datasets)
    predict_test_data(rfc, test_dataset)

    # model 3 logistic regression: 
    name = 'logistic regression'
    lr = get_logistic_regression(training_dataset, training_labels)
    display_metrics_for_validation(lr, name, *datasets)
    predict_test_data(lr, test_dataset)
    
    # model 4 gradient boosting:
    name = 'gradient boosting'
    gb = get_gradient_boosing_classifier(training_dataset, training_labels)
    display_metrics_for_validation(gb, name, *datasets)
    predict_test_data(gb, test_dataset)
    
    
    
    sys.exit(0)