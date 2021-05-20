from sklearn.metrics import (
                                f1_score, 
                                recall_score,
                                roc_auc_score,
                                plot_roc_curve,
                                roc_auc_score,
                                accuracy_score,
                                plot_confusion_matrix,
                            )
import matplotlib.pyplot as plt

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
    
def display_metrics_for_validation(
                                    model,
                                    model_name,
                                    validation_dataset,
                                    validation_labels
                                   ):
    print(model_name)
    print("Metrics for validation dataset:")
    display_metrics(model_name + " validation", *calculate_metrics(model, validation_dataset, validation_labels))