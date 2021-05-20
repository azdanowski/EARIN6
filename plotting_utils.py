import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def plot_p_r_curves(model1, model2, model1_name, model2_name, data, labels):
    """Used to plot recall-precision curves for different models
    for comparison.
    """
    pred1 = model1.predict(data)
    pred2 = model2.predict(data)

    recall1, precision1, _ = precision_recall_curve(labels, pred1)
    recall2, precision2, _ = precision_recall_curve(labels, pred2)

    plt.plot(recall1, precision1, label=model1_name)
    plt.plot(recall2, precision2, label=model2_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
