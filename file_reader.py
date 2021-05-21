import pandas as pd

def read_training_data():
    # read training file:
    dataset = pd.read_csv('data/arcene_train.data', sep='\s+', header=None)
    # validate file: 
    assert dataset.sum().sum() == 70726744.0    # checksum 
    assert dataset.isna().sum().sum() == 0      # no NaN values
    return dataset

def read_training_labels(training_dataset):
# load training labels:
    dataset_labels = pd.read_csv('data/arcene_train.labels', header=None)
    # validate file:
    assert dataset_labels.shape[0] == training_dataset.shape[0]  # matrix dimensions match
    assert dataset_labels.isna().sum().sum() == 0       # no NaN values
    return dataset_labels

def read_validation_dataset():
    # load dataset
    validation_dataset = pd.read_csv('data/arcene_valid.data', sep='\s+', header=None)
    # validate file:
    assert validation_dataset.sum().sum() == 71410108.0     # checksum OK
    assert validation_dataset.isna().sum().sum() == 0       # no Nan values
    return validation_dataset

def read_validation_labels(validation_dataset):
    # load dataset
    validation_labels = pd.read_csv('data/arcene_valid.labels', header=None)
    # validate file:
    assert validation_labels.shape[0] == validation_dataset.shape[0]
    assert validation_labels.isna().sum().sum() == 0
    return validation_labels

def read_test_dataset():
    test_dataset = pd.read_csv('data/arcene_test.data', sep='\s+', header=None)
    assert test_dataset.sum().sum() == 493023349.0
    assert test_dataset.isna().sum().sum() == 0
    return test_dataset