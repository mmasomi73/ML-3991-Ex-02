import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from anomatools.models import SSDO
from sklearn.decomposition import PCA

from Evaluator import Evaluator
from OutputWriter import OutputWriter


class SSDOHandler:
    path_to_plt = '../outs/charts/ssdo/'
    contamination = 0.3
    k = 10

    st_tr_time = []
    en_tr_time = []
    st_te_time = []
    en_te_time = []

    def __init__(self, dataCollector):
        self.dataCollector = dataCollector
        if not os.path.exists(self.path_to_plt):
            os.mkdir(self.path_to_plt)
            os.mkdir(self.path_to_plt + 'anom')
            os.mkdir(self.path_to_plt + 'chart')

    def findAnomalies(self, saveChart=False, saveEvaluation=False):
        predicted_outlier = []
        list_of_df = self.dataCollector.getWithAnomaly()
        aida = 0
        for df in list_of_df:
            aida += 1
            if df.shape[0] > 0:
                clf = SSDO()
                data = df.drop(['anomaly', 'changepoint'], axis=1)
                self.st_tr_time.append(datetime.datetime.now().timestamp())
                clf.fit(data)
                self.en_tr_time.append(datetime.datetime.now().timestamp())
                self.st_te_time.append(datetime.datetime.now().timestamp())
                prediction = pd.Series(clf.predict(data.to_numpy()), index=df.index) \
                    .rolling(5) \
                    .median() \
                    .fillna(0).replace(-1, 0)
                self.en_te_time.append(datetime.datetime.now().timestamp())
                # predicted outliers saving
                predicted_outlier.append(prediction)
                df['ssdo_anomaly'] = prediction

        print(len(predicted_outlier))
        true_outlier = [df.anomaly for df in list_of_df]
        if saveChart:
            for i in range(len(predicted_outlier)):
                plt.figure()

                plt.rcParams["font.family"] = "Times New Roman"
                csfont = {'fontname': 'Times New Roman'}
                plt.xlabel('Time', **csfont)
                plt.ylabel('Value', **csfont)
                plt.title('SSDO On File [{}]'.format(i + 1), **csfont)

                predicted_outlier[i].plot(figsize=(12, 6), label='predictions', marker='o', markersize=5)
                true_outlier[i].plot(marker='o', markersize=2)

                # data = list_of_df[i]
                # plt.scatter(x=data[data['forest_anomaly'] == data['anomaly']].index,
                #             y=data[data['forest_anomaly'] == data['anomaly']]['anomaly'], label='True Prediction'
                #             , c='g', zorder=4)
                # plt.scatter(x=data[data['forest_anomaly'] != data['anomaly']].index,
                #             y=data[data['forest_anomaly'] != data['anomaly']]['anomaly'], label='False Prediction'
                #             , c='r', zorder=5)
                plt.legend(loc='upper right')
                plt.savefig(self.path_to_plt + 'anom/ssdo-pre-{}.png'.format(i + 1), format='png')
                print('Chart {} is Generated'.format(i + 1))
                plt.clf()
                plt.close('all')
        if saveChart:
            ts = 1
            for df in list_of_df:
                data = df.drop(['anomaly', 'changepoint'], axis=1)
                pc = PCA(n_components=2).fit_transform(data)
                df[['X', 'Y']] = pc
                plt.figure()
                sb.set(font='Times New Roman')
                sns = sb.scatterplot(data=df, x='X', y='Y', hue='ssdo_anomaly', palette='bright')
                sns.set_title('The Anomaly Detected By SSDO, File {}'.format(ts))
                sns.figure.savefig(self.path_to_plt + 'chart/chart-{}.png'.format(ts))
                plt.close('all')
                print('The Chart of  File {} is Generated.'.format(ts))
                ts += 1
        if saveEvaluation:
            evaluator = Evaluator(true_outlier, predicted_outlier, metric='binary', numenta_time='30 sec')
            metrics = evaluator.getConfusionMetrics()
            TP = metrics['TP']
            TN = metrics['TN']
            FP = metrics['FP']
            FN = metrics['FN']
            print(f'\t False Alarm Rate: {round(FP / (FP + TN) * 100, 2)} %')
            print(f'\t Missing Alarm Rate: {round(FN / (FN + TP) * 100, 2)} %')
            print(f'\t Accuracy Rate: {round((TP + TN) / (TP + TN + FN + TP) * 100, 2)} %')

            trainTime = np.array(self.en_tr_time).sum() - np.array(self.st_tr_time).sum()
            testTime = np.array(self.en_te_time).sum() - np.array(self.st_te_time).sum()
            print(f'\t Train Time {round(trainTime, 2)}s')
            print(f'\t Test Time {round(testTime, 2)}s')

            data = {'far': round(FP / (FP + TN) * 100, 2),
                    'mar': round(FN / (FN + TP) * 100, 2),
                    'acc': round((TP + TN) / (TP + TN + FN + TP) * 100, 2),
                    'tr': trainTime,
                    'te': testTime,
                    'tp': TP,
                    'tn': TN,
                    'fp': FP,
                    'fn': FN}
            output = OutputWriter(self.path_to_plt, 'SSDO', data)
            output.write()

    def trainTime(self):
        return np.array(self.en_tr_time).sum() - np.array(self.st_tr_time).sum()

    def testTime(self):
        return np.array(self.en_te_time).sum() - np.array(self.st_te_time).sum()
