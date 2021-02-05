import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree


class KNNOHandler:
    path_to_plt = '../outs/charts/knno/'
    contamination = 0.3
    k = 10

    def __init__(self, dataCollector):
        self.dataCollector = dataCollector

    def findAnomalies(self, saveChart=False):
        predicted_outlier = []
        list_of_df = self.dataCollector.getWithAnomaly()
        for df in list_of_df:
            if df.shape[0] > 0:
                # clf = SSkNNO()
                data = df.drop(['anomaly', 'changepoint'], axis=1)

                prediction = pd.Series(self.fit_predict(data.to_numpy()), index=df.index) \
                    .rolling(5) \
                    .median() \
                    .fillna(0)
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

    def fit_predict(self, Xt):
        tree = BallTree(Xt, leaf_size=16, metric='euclidean')
        D, _ = tree.query(Xt, k=self.k + 1)

        # predict
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
        return labels

    def _squashing_function(self, x, p):
        return 1.0 - np.exp(np.log(0.5) * np.power(x / p, 2))