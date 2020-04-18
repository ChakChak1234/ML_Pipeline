import os, sys
import numpy as np
from .binary_clf_curve import _binary_clf_curve

class calc_metrics:
    """
    This function uses the following metrics to assess the training and test sets.

    Precision:
    The ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample
    that is negative.
    The best value is 1 and the worst value is 0.

    Recall:
    The ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.
    The best value is 1 and the worst value is 0.

    AUPRC:
    The Area Under Precision Recall Curve calculates the average of the weighted mean of precision
    at each threshold, using recall from the previous iteration as the weight.

    ``Sum_n(R_n - R_n-1)*P_n``
    P_n and R_n are the precision and recall at the nth threshold.

    AUROC:
    The Area Under Receiver Operating Characteristic Curve calculates the model's capability to distinguish
    between classes

    :return:
    """
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def confusion_matrix(self):
        actual = self.actual
        predicted = self.predicted
        tp = 0
        fp = 0
        tn = 0
        fn = 0

        for i in range(len(actual)):
            if int(actual[i]) == 1:
                if actual[i] == predicted[i]:
                    tp += 1
                elif actual[i] != predicted[i]:
                    fn += 1
            elif int(actual[i]) == 0:
                if actual[i] == predicted[i]:
                    tn += 1
                elif actual[i] != predicted[i]:
                    fp += 1

        # Precision
        precision = tp / (tp + fp)

        # Recall
        recall = tp / (tp + fn)

        return precision, recall

    def auprc(self):
        actual = self.actual
        predicted = self.predicted

        k = len(predicted)
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        auprc = score / min(len(actual), k)

        return auprc

    def auroc(self):
        actual = self.actual
        predicted = self.predicted

        tps, fps, _ = _binary_clf_curve(actual, predicted)
        # convert count to rate
        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # compute AUC using the trapezoidal rule;
        # appending an extra 0 is just to ensure the length matches
        zero = np.array([0])
        tpr_diff = np.hstack((np.diff(tpr), zero))
        fpr_diff = np.hstack((np.diff(fpr), zero))
        auroc = np.dot(tpr, fpr_diff) + np.dot(tpr_diff, fpr_diff) / 2

        return auroc

    def return_metrics(self):
        precision, recall = self.confusion_matrix()

        auprc = self.auprc()
        auroc = self.auroc()

        return precision, recall, auprc, auroc