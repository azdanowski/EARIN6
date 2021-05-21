import sys
import json
from file_reader import *
from classifiers import *
from metrics import *
from utils import *
from sklearn.model_selection import train_test_split

def train_test_split_assert(training_dataset, training_labels, train):
    # train must be float (0 : 1) to represent proportion
    # or int to represent number of samples
    x_train, x_test, y_train, y_test = train_test_split(
        training_dataset,
        training_labels,
        train_size=train,
        random_state=0
        )
    # validate dimensions
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    # validate dimensions between split and dataset
    assert x_train.shape[0] + x_test.shape[0] == training_dataset.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == training_labels.shape[0]
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':

    # preparing dataset
    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    training = (training_dataset, training_labels)
    
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)
    validation = (validation_dataset, validation_labels)
    
    test_dataset = read_test_dataset()
    
    t = 0.8 # train to test proportion,
            # example: 0.8 for 80-20 training-test split
    x_train, x_test, y_train, y_test = train_test_split_assert(*training, t)
   

    metrics_comparison = {}
    metrics_comparison["metrics"] = ["f1", "recall", "roc_auc", "accuracy"]


    model_names = [
                    'linear support vector',
                    'random forest',
                    'logistic regression',
                    'gradient boosting',
                    'k neighbors'
                  ]

    for name in model_names:
        model = get_trained_model(name, x_train, y_train)
        message = "train-test-split"
        metrics_comparison[name] = calculate_and_display_metrics(
            model, message, name, x_test, y_test)
        predict_test_data(model, test_dataset)



    # for name in model_names:
    #     model = get_trained_model(name, training_dataset, training_labels)
    #     message = 'validation'
    #     metrics_comparison[name] = calculate_and_display_metrics(
    #         model, message, name, *validation)
    #     predict_test_data(model, test_dataset)



    # # model 1: lsvc
    # name = "linear support vector"
    # lsvc = get_trained_model(name, training_dataset, training_labels)
    # metrics_comparison[name] = display_metrics_for_validation(
    #     lsvc, name, *validation)
    # predict_test_data(lsvc, test_dataset)

    # # model 2: random forest
    # name = "random forest"
    # rfc = get_trained_model(name, training_dataset, training_labels)
    # metrics_comparison[name] = display_metrics_for_validation(
    #     rfc, name, *validation)
    # predict_test_data(rfc, test_dataset)

    # # model 3 logistic regression:
    # name = 'logistic regression'
    # lr = get_trained_model(name, training_dataset, training_labels)
    # metrics_comparison[name] = display_metrics_for_validation(
    #     lr, name, *validation)
    # predict_test_data(lr, test_dataset)

    # # model 4 gradient boosting:
    # name = 'gradient boosting'
    # gb = get_trained_model(name, training_dataset, training_labels)
    # metrics_comparison[name] = display_metrics_for_validation(
    #     gb, name, *validation)
    # predict_test_data(gb, test_dataset)

    # # model 5 k neighbors:
    # name = 'k neigbors'
    # kn = get_trained_model(name, training_dataset, training_labels)
    # metrics_comparison[name] = display_metrics_for_validation(
    #     kn, name, *validation)
    # predict_test_data(kn, test_dataset)

    # compare accuracy of all models:
    print("the most accurate model:")
    print(max_val(metrics_comparison))
    # store accuracy in file
    with open('results/metrics_table.json', 'w') as file:
        json.dump(metrics_comparison, file)

    sys.exit(0)
