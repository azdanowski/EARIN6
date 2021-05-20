import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    datasets = (training_dataset, training_labels, validation_dataset, validation_labels)

    
    # model 1: lsvc
    name = "linear support vector"
    lsvc = get_linear_support_vector_classifier(training_dataset, training_labels)
    display_metrics_for_training_and_validation(lsvc, name, *datasets)
    predict_test_data(lsvc, test_dataset)



    # model 2: random forest
    name = "random forest"
    rfc = get_random_forest_classifier(training_dataset, training_labels)
    display_metrics_for_training_and_validation(rfc, name, *datasets)
    predict_test_data(rfc, test_dataset)

    # model 3: 



    sys.exit(0)