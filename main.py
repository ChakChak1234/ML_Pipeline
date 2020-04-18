import os, sys, shutil
import pandas as pd
import numpy as np
from random import seed
from random import randrange

from data_split.main import DataSplitProcedures
from braf.main import BiasedRandomForestClassifier
from metrics.main import calc_metrics
from plots.main import plot_roc_auc, plot_prec_recall

class BRAF_PIPELINE:
    """
    Biased Random Forest Pipeline
    """
    def __init__(self, data, s, p, K):
        self.data = data
        self.s = s
        self.p = p
        self.K = K
        self.col_names = data.columns.tolist()
        self.n_folds = 10
        self.test_scenario = dict()
        self.test_scenario['scenario'] = {}
        self.test_scenario['scenario']['train_test'] = {}
        self.test_scenario['scenario']['kfold_crossval'] = {}

        self.results = dict()
        self.results['scenario'] = {}
        self.results['scenario']['train_test'] = {}
        self.results['scenario']['train_test']['scores'] = {}
        self.results['scenario']['train_test']['values'] = {}
        self.results['scenario']['kfold_crossval'] = {}
        self.results['scenario']['kfold_crossval']['scores'] = {}
        self.results['scenario']['kfold_crossval']['values'] = {}

    def preprocess(self):
        """
        This function processes the data before running the braf
        :return:
        """
        # data clean

        # missing data imputation
        return self.data

    def split_data(self):
        """
        This function splits data
        :return:
        """
        self.DataSplitProcedures = DataSplitProcedures(df = self.data, n_folds = 10)
        ds_proc = self.DataSplitProcedures
        train_test, kfold_crossval = ds_proc.run()

        self.test_scenario['scenario']['train_test'] = train_test
        self.test_scenario['scenario']['kfold_crossval'] = kfold_crossval

        return self.test_scenario['scenario']['train_test'], self.test_scenario['scenario']['kfold_crossval']

    def run_model(self):
        """
        This function fits and trains the data
        :return:
        """
        scenarios_data  = self.split_data()
        scenario = ['train_test', 'kfold_crossval']

        count = 0
        for data in scenarios_data:
            braf = BiasedRandomForestClassifier(df = data[0], columns = self.col_names,
                                                s = self.s, p = self.p, K = self.K)
            values = braf.combine_forests()
            #self.results['scenario'][scenario[count]]['scores'] = scores
            self.results['scenario'][scenario[count]]['values'] = values
            count += 1

        return self.results

    def produce_results(self):
        """
        This function outputs results
        :return:
        """
        self.run_model()

        results = self.results['scenario']

        for scenarios in results.keys():
            for values in results[scenarios]['values'].keys():
                vals = results[scenarios]['values'][values]['braf']
                actual = []
                predicted = []
                for a, p in zip(vals['actual'], vals['predicted']):
                    actual.append(a)
                    predicted.append(p)
                precision, recall, auprc, auroc = calc_metrics(a, p).return_metrics()
                print(precision, recall, auprc, auroc)

    def produce_plots(self):
        """
        This function produces and plots the ROC Curve and Precision-Recall Curve
        :return:
        """

        results = self.results['scenario']

        for scenarios in results.keys():
            for values in results[scenarios]['values'].keys():
                vals = results[scenarios]['values'][values]['braf']
                actual = []
                predicted = []
                for a, p in zip(vals['actual'], vals['predicted']):
                    actual.extend(a)
                    predicted.extend(p)
                roc_plot = plot_roc_auc(actual, predicted)
                roc_plot.savefig('Receiver Operator Curve ({}).png'.format(scenarios))
                prc_plot = plot_prec_recall(actual, predicted)
                prc_plot.savefig('Precision-Recall Curve ({}).png'.format(scenarios))

    def run(self):
        """
        This runs everything
        :return:
        """
        self.run_model()
        self.produce_results()
        self.produce_plots()


if __name__ == "__main__":
    filename = os.path.abspath(os.path.join(os.getcwd(), 'Data', 'diabetes.csv'))
    dataset = pd.read_csv(filename, sep=',', header=0)

    results = BRAF_PIPELINE(data = dataset, s = 100, p = .5, K = 10).run()
    print(results)