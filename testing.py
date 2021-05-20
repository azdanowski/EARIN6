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

from main import *  # xDDD
from plotting_utils import *

class ReaaaalySmart:

    def __init__(self):

        # preparing dataset
        self.training_dataset = read_training_data()
        self.training_labels = read_training_labels(self.training_dataset)
        self.validation_dataset = read_validation_dataset()
        self.validation_labels = read_validation_labels(
            self.validation_dataset)
        self.test_dataset = read_test_dataset()
        # group together training and validation
        self.datasets = (self.training_dataset, self.training_labels,
                    self.validation_dataset, self.validation_labels)

    def do_some_testing(self):
        model_names = []
        models = []

        lsvc1 = LinearSVC(random_state=0, tol=1e-6, class_weight='balanced')
        lsvc1.fit(self.training_dataset, self.training_labels.ravel())
        models.append(lsvc1)
        model_names.append("lsvc1")

        lsvc2 = LinearSVC(random_state=0, tol=1e-6, class_weight='balanced')
        lsvc2.fit(self.training_dataset, self.training_labels.ravel())
        models.append(lsvc2)
        model_names.append("lsvc2")

        prc = PrecisionRecallCurves(models, model_names, self.training_dataset,
            self.training_labels)
        prc.plot()


if __name__ == '__main__':

    tests = ReaaaalySmart()
    tests.do_some_testing()
