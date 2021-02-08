import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.neighbors import BallTree
from Evaluator import Evaluator
import numpy as np

from OutputWriter import OutputWriter


class KNNOHandler:
    path_to_plt = '../outs/charts/knno/'
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
        for df in list_of_df:
            if df.shape[0] > 0:
                # clf = SSkNNO()
                data = df.drop(['anomaly', 'changepoint'], axis=1)
                self.st_tr_time.append(datetime.datetime.now().timestamp())
                prediction = pd.Series(self.fit_predict(data.to_numpy()), index=df.index) \
                    .rolling(5) \
                    .median() \
                    .fillna(0)
                self.en_tr_time.append(datetime.datetime.now().timestamp())
                # predicted outliers saving
                predicted_outlier.append(prediction)
                df['knno_anomaly'] = prediction

        true_outlier = [df.anomaly for df in list_of_df]
        if saveChart:
            for i in range(len(predicted_outlier)):
                plt.figure()

                plt.rcParams["font.family"] = "Times New Roman"
                csfont = {'fontname': 'Times New Roman'}
                plt.xlabel('Time', **csfont)
                plt.ylabel('Value', **csfont)
                plt.title('KNNO On File [{}]'.format(i + 1), **csfont)

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
                plt.savefig(self.path_to_plt + 'anom/knno-pre-{}.png'.format(i + 1), format='png')
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
                sns = sb.scatterplot(data=df, x='X', y='Y', hue='knno_anomaly', palette='bright')
                sns.set_title('The Anomaly Detected By KNNO, File {}'.format(ts))
                sns.set_title('The Anomaly Detected By KNNO, File {}'.format(ts))
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
            print(f'\t Train & Test Time {round(trainTime, 2)}s')

            data = {'far': round(FP / (FP + TN) * 100, 2),
                    'mar': round(FN / (FN + TP) * 100, 2),
                    'acc': round((TP + TN) / (TP + TN + FN + TP) * 100, 2),
                    'tr': trainTime,
                    'te': 0,
                    'tp': TP,
                    'tn': TN,
                    'fp': FP,
                    'fn': FN}
            output = OutputWriter(self.path_to_plt, 'KNNO', data)
            output.write()

    def trainTime(self):
        return np.array(self.en_tr_time).sum() - np.array(self.st_tr_time).sum()

    def fit_predict(self, Xt):
        self.st_tr_time = datetime.datetime.now().timestamp()
        tree = BallTree(Xt, leaf_size=16, metric='euclidean')
        D, _ = tree.query(Xt, k=self.k + 1)
        self.en_tr_time = datetime.datetime.now().timestamp()
        # predict
        self.st_te_time = datetime.datetime.now().timestamp()
        outlier_scores = D[:, -1].flatten()
        gamma = np.percentile(
            outlier_scores, int(100 * (1.0 - self.contamination))) + 1e-10
        yt_scores = self._squashing_function(outlier_scores, gamma)
        labels = []
        for y in yt_scores:
            if y <= self.contamination:
                labels.append(1)
            else:
                labels.append(0)
        self.en_te_time = datetime.datetime.now().timestamp()
        return labels

    def _squashing_function(self, x, p):
        return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))
