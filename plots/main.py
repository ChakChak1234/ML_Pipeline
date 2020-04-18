import matplotlib.pyplot as plt
from metrics.roc_curve import roc_curve
from metrics.precision_recall_curve import precision_recall_curve
from metrics.auc import auc


def plot_roc_auc(actual, predicted):
    fpr, tpr, threshold = roc_curve(actual, predicted)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    return plt


def plot_prec_recall(actual, predicted):
    precisions, recalls, thresholds = precision_recall_curve(actual, predicted)

    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])

    return plt
