import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA


class RobustCovHandler:
    path_to_plt = '../outs/charts/rocov/'

    def __init__(self, dataCollector):
        self.dataCollector = dataCollector
        if not os.path.isfile(self.path_to_plt):
            os.mkdir(self.path_to_plt)
            os.mkdir(self.path_to_plt + 'anom')
            os.mkdir(self.path_to_plt + 'chart')

    def findAnomalies(self, saveChart=False):
        outliers_fraction = 0.15
        clf = EllipticEnvelope(contamination=outliers_fraction)
        predicted_outlier = []
        list_of_df = self.dataCollector.getWithAnomaly()
        for df in list_of_df:
            if df.shape[0] > 0:
                data = df.drop(['anomaly', 'changepoint'], axis=1)
                prediction = pd.Series(clf.fit_predict(data) * -1, index=df.index) \
                    .rolling(5) \
                    .median() \
                    .fillna(0).replace(-1, 0)
                # predicted outliers saving
                predicted_outlier.append(prediction)
                df['rocov_anomaly'] = prediction
        true_outlier = [df.anomaly for df in list_of_df]
        if saveChart:
            for i in range(len(predicted_outlier)):
                plt.figure()

                plt.rcParams["font.family"] = "Times New Roman"
                csfont = {'fontname': 'Times New Roman'}
                plt.xlabel('Time', **csfont)
                plt.ylabel('Value', **csfont)
                plt.title('Robust covariance On File [{}]'.format(i + 1), **csfont)

                predicted_outlier[i].plot(figsize=(12, 6), label='predictions', marker='o', markersize=5)
                true_outlier[i].plot(marker='o', markersize=2)

                # data = list_of_df[i]
                # plt.scatter(x=data[data['rocov_anomaly'] == data['anomaly']].index,
                #             y=data[data['rocov_anomaly'] == data['anomaly']]['anomaly'], label='True Prediction'
                #             , c='g', zorder=4)
                # plt.scatter(x=data[data['rocov_anomaly'] != data['anomaly']].index,
                #             y=data[data['rocov_anomaly'] != data['anomaly']]['anomaly'], label='False Prediction'
                #             , c='r', zorder=5)
                plt.legend(loc='upper right')
                plt.savefig(self.path_to_plt + 'anom/rocov-pre-{}.png'.format(i + 1), format='png')
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
                sns = sb.scatterplot(data=df, x='X', y='Y', hue='rocov_anomaly', palette='bright')
                sns.set_title('The Anomaly Detected By Robust covariance, File {}'.format(ts))
                sns.figure.savefig(self.path_to_plt + 'chart/chart-{}.png'.format(ts))
                plt.close('all')
                print('The Chart of  File {} is Generated.'.format(ts))
                ts += 1
