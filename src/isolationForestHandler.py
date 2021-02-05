import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt


class IsolationForestHandler:
    path_to_plt = '../outs/charts/isolation_forest/'

    def __init__(self, dataCollector):
        self.dataCollector = dataCollector

    def findAnomalies(self):
        clf = IsolationForest(random_state=0, n_jobs=-1)
        predicted_outlier = []
        list_of_df = self.dataCollector.getWithAnomaly()
        for df in list_of_df:
            if df.shape[0] > 0:
                data = df.drop(['anomaly', 'changepoint'], axis=1)
                prediction = pd.Series(clf.fit_predict(data)* -1, index=df.index) \
                    .rolling(5) \
                    .median() \
                    .fillna(0).replace(-1, 0)

                # predicted outliers saving
                predicted_outlier.append(prediction)
                df['forest_anomaly'] = prediction

        true_outlier = [df.anomaly for df in list_of_df]
        for i in range(len(predicted_outlier)):
            plt.figure()

            plt.rcParams["font.family"] = "Times New Roman"
            csfont = {'fontname': 'Times New Roman'}
            plt.xlabel('Time', **csfont)
            plt.ylabel('Value', **csfont)
            plt.title('Isolation Forest On File [{}]'.format(i+1), **csfont)

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
            plt.savefig(self.path_to_plt + 'iso_for-pre-{}.png'.format(i+1), format='png')
            print('Chart {} is Generated'.format(i+1))
            plt.clf()
            plt.close('all')
