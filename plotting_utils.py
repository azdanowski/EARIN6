import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve as prc


class PrecisionRecallComparisonDisplay:
    """Class to store matplotlib objects to visualize precision recall
    curves comparison across multiple models.
    """

    def __init__(self, precisions, recalls, model_names):
        # some asserts for fun ;)
        assert len(precisions) == len(recalls)
        assert len(recalls) == len(model_names)

        # we can rest assured of the lengths and keep initializing ;)
        self.precisions = precisions
        self.recalls = recalls
        self.model_names = model_names

        self.fig = plt.figure()
        self.axes = [] * self.length

        self._add_axes()
        self._plot_on_axes()

    def _add_axes(self):
        """Helper method for the initialization
        of PrecisionRecallComparisonDisplay
        to be more readable.

        It adds axes to self.fig, one for each data series.
        """

        for i in range(self.length):
            self.axes.append(self.fig.add_axes([0.1, 0.3, 0.8, 0.8]))

    def _plot_on_axes(self):
        """Helper method for the initialization
        of PrecisionRecallComparisonDisplay
        to be more readable.

        It plots aggregated precision recall curves by means
        of each axis.
        """

        for i, ax in enumerate(self.axes):
            ax.plot(self.precisions[i], self.recalls[i])

    def plot(self):
        """Method for displaying the stored plot."""

        plt.show()


class PrecisionRecallCurves:
    """Class to store precision recall curves
    based on its initialization parameters.
    Also, a method is provided to display the plot of aggregated curves.
    """

    def __init__(self, models, data, labels, model_names):
        """
        models - models to predict with
        data   - data sets for the models (separate data for each model)
        labels - labels of the data (separate labels for each model)
        model_names - modale names

        All the parameters have to be provided in respective order.
        """

        # asserts to assure the parameters sizes are equal:
        assert len(models) == len(data)
        assert len(data) == len(labels)
        assert len(labels) == len(model_names)

        # assurance completed, now it is safe to initialize:
        self.models = models
        self.data = data
        self.labels = labels
        self.model_names = model_names

        self.recalls = self._init_rp(len(self.models))
        self.precisions = self.init_rp(len(self.models))

    def _init_rp(self, length):
        """Helper method to initialize self.recalls and self.precisions
        in a static manner, based on the length of __init__ parameters.
        """

        return [] * length

    def _predict(self):
        """Method to predict on the data fed to the models."""

        for i, model in enumerate(self.models):
            pred = model.predict(self.data[i])
            self.recalls[i], self.precisions[i], _ = prc(self.labels[i], pred)

    def get_curves(self):
        """Returns recall and precision curves."""

        self._predict()

        return self.recalls, self.precisions

    def plot(self):
        """Method to plot aggregated curves."""

        self._predict()

        figures = PrecisionRecallComparisonDisplay(
            self.precisions, self.recalls, self.model_names)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        figures.plot()
