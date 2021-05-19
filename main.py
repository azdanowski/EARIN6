import sys
import pandas as pd
from sklearn.model_selection import train_test_split

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



if __name__ == '__main__':

    training_dataset = read_training_data()
    training_labels = read_training_labels(training_dataset)
    x_train, x_test, y_train, y_test = split_data_train_test(training_dataset, training_labels)
    validation_dataset = read_validation_dataset()
    validation_labels = read_validation_labels(validation_dataset)

    






    sys.exit(0)