import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
                                f1_score, 
                                recall_score,
                                roc_auc_score,
                                plot_roc_curve,
                                roc_auc_score,
                                accuracy_score,
                                plot_confusion_matrix,
                            )
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def read_training_data():
    # read training file:
    dataset = pd.read_csv('arcene_train.data', sep='\s+', header=None)
    # validate file: 
    assert dataset.sum().sum() == 70726744.0    # checksum 
    assert dataset.isna().sum().sum() == 0      # no NaN values
    return dataset

def read_training_labels(training_dataset):
# load training labels:
    dataset_labels = pd.read_csv('arcene_train.labels', header=None)
    # validate file:
    assert dataset_labels.shape[0] == training_dataset.shape[0]  # matrix dimensions match
    assert dataset_labels.isna().sum().sum() == 0       # no NaN values
    return dataset_labels

def read_validation_dataset():
    # load dataset
    validation_dataset = pd.read_csv('arcene_valid.data', sep='\s+', header=None)
    # validate file:
    assert validation_dataset.sum().sum() == 71410108.0     # checksum OK
    assert validation_dataset.isna().sum().sum() == 0       # no Nan values
    return validation_dataset

def read_validation_labels(validation_dataset):
    # load dataset
    validation_labels = pd.read_csv('arcene_valid.labels', header=None)
    # validate file:
    assert validation_labels.shape[0] == validation_dataset.shape[0]
    assert validation_labels.isna().sum().sum() == 0
    return validation_labels

def read_test_dataset():
    test_dataset = pd.read_csv('arcene_test.data', sep='\s+', header=None)
    return test_dataset

def split_data_train_test(training_dataset, training_labels):
    # split into train (0.8), test (0.2)
    x_train, x_test, y_train, y_test = train_test_split(
        training_dataset,
        training_labels,
        test_size=0.2,
        random_state=0
        )
    # validate dimensions
    assert x_train.shape[0] == y_train.shape[0]
    assert x_test.shape[0] == y_test.shape[0]
    # validate dimensions between split and dataset
    assert x_train.shape[0] + x_test.shape[0] == training_dataset.shape[0]
    assert y_train.shape[0] + y_test.shape[0] == training_labels.shape[0]
    return x_train, x_test, y_train, y_test

def display_metrics(model_name, f1, recall, roc_auc, accuracy, plot_c_m, plot_r_c):
    
    # display metrics
    print(model_name)
    print(f'    F1 score - {f1*100:.2f}%.')
    print(f'    Recall score - {recall*100:.2f}%.')
    print(f'    ROC AUC score - {roc_auc[0]*100:.2f}%.')
    print(f'    Accuracy score - {accuracy*100:.2f}%.')
    
    plot_c_m.ax_.set_title(model_name + " confusion matrix")
    plot_r_c.ax_.set_title(model_name + " roc curve")
    plt.show()
    

def calculate_metrics(model, data, labels):
    # make prediction 
    predicted_labels = model.predict(data)
    # calculate metrics
    f1 = f1_score(labels, predicted_labels)
    recall = recall_score(labels, predicted_labels)
    roc_auc = roc_auc_score(labels, predicted_labels),
    accuracy = accuracy_score(labels, predicted_labels)
    plot_c_m = plot_confusion_matrix(model, data, labels, normalize='true')
    plot_r_c = plot_roc_curve(model, data, labels)
    return f1, recall, roc_auc, accuracy, plot_c_m, plot_r_c

def get_linear_support_vector_classifier(x_train, y_train):
    # initialize
    lsvc = LinearSVC(random_state=0, tol=1e-6, class_weight='balanced')
    # fit model, flatten y_train to 1 row
    lsvc.fit(x_train, y_train.values.ravel())
    return lsvc

def get_random_forest_classifier(x_train, y_train):
    rfc = RandomForestClassifier(random_state=0)
    rfc.fit(x_train, y_train.values.ravel())
    return rfc
   
def predict_test_data(model, test_dataset):
    predictions = model.predict(test_dataset)
    negative = 0
    positive = 0
    for i in range(len(predictions)):
        if predictions[i] < 0:
            negative+=1
        else:
            positive+=1
    print(f"    positive: {positive}" + " expected: 310")
    print(f"    negative: {negative}" + " expected: 390")   
    
def display_metrics_for_training_and_validation(
                                                model,
                                                model_name,
                                                training_dataset,
                                                training_labels,
                                                validation_dataset,
                                                validation_labels
                                                ):
    print(model_name)
    print("Metrics for training dataset:")
    display_metrics(model_name + " training", *calculate_metrics(model, training_dataset, training_labels))
    print("Metrics for validation dataset:")
    display_metrics(model_name + " validation", *calculate_metrics(model, validation_dataset, validation_labels))
    

if __name__ == '__main__':
    
    # preparing dataset
    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    x_train, x_test, y_train, y_test = split_data_train_test(training_dataset, training_labels)
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)
    test_dataset = read_test_dataset()
    # group together training and validation
    datasets = (training_dataset, training_labels, validation_dataset, validation_labels)

    
    # model 1: lsvc
    name = "linear support vector"
    lsvc = get_linear_support_vector_classifier(x_train, y_train)
    display_metrics_for_training_and_validation(lsvc, name, *datasets)
    predict_test_data(lsvc, test_dataset)



    # model 2: random forest
    name = "random forest"
    rfc = get_random_forest_classifier(x_train, y_train)
    display_metrics_for_training_and_validation(rfc, name, *datasets)
    predict_test_data(rfc, test_dataset)

   # rfc = get_random_forest_classifier(x_train, y_train)
    
   # metrics(rfc, x_test, y_test)
   # metrics(lsvc, validation_dataset, validation_labels)



    sys.exit(0)