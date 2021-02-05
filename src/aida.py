import datetime
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from DataCollector import DataCollector
from isolationForestHandler import IsolationForestHandler

dc = DataCollector(False)
# data = dc.dataSetReader()

exec_chart_sec = False
if exec_chart_sec:
    path_to_plt = '../outs/charts/main/'
    list_of_df = dc.getWithAnomaly()
    ts = 1
    for df in list_of_df:
        if df.shape[0] > 0:
            plt.rcParams["font.family"] = "Times New Roman"
            csfont = {'fontname': 'Times New Roman'}
            df.plot(figsize=(12, 9), subplots=True)
            plt.xlabel('Time', **csfont)
            plt.ylabel('Value', **csfont)
            plt.title('Signals[{}]'.format(ts), **csfont)
            plt.legend(loc='upper right')
            plt.savefig(path_to_plt + 'main-{}.png'.format(ts), format='png')
            plt.close('all')
            print('The Chart of  File {} is Generated.'.format(ts))
            ts += 1


exec_chart_anom = False
if exec_chart_anom:
    path_to_plt = '../outs/charts/anomalies/'
    list_of_df = dc.getWithAnomaly()
    ts = 1
    data = pd.DataFrame([])
    for df in list_of_df:
        if df.shape[0] > 0:
            pc = PCA(n_components=2).fit_transform(df)
            df[['X', 'Y']] = pc
            plt.figure()
            sb.set(font='Times New Roman')
            sns = sb.scatterplot(data=df, x='X', y='Y', hue='anomaly', palette='bright')
            sns.set_title('The Anomaly Chart of  File {}'.format(ts))
            sns.figure.savefig(path_to_plt + 'anom-{}.png'.format(ts))
            plt.close('all')
            print('The Chart of  File {} is Generated.'.format(ts))
            ts += 1


# isofh = IsolationForestHandler(dc)
# isofh.findAnomalies()


