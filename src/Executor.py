import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA

from DataCollector import DataCollector
from KNNOHandler import KNNOHandler
from LOFHandler import LOFHandler
from OneClassSVMHandler import OneClassSVMHandler
from RobustCovHandler import RobustCovHandler
from SSDOHandler import SSDOHandler
from SSkNNOHandler import SSkNNOHandler
from isolationForestHandler import IsolationForestHandler


class Executor:
    draw_chart = False
    write_file = False
    exec_chart_sec = False
    exec_chart_anom = False

    def __init__(self):
        self.dc = DataCollector(False)

    def drawCahrts(self):
        exec_chart_sec = False
        if exec_chart_sec:
            path_to_plt = '../outs/charts/main/'
            list_of_df = self.dc.getWithAnomaly()
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
            list_of_df = self.dc.getWithAnomaly()
            ts = 1
            for df in list_of_df:
                if df.shape[0] > 0:
                    data = df.drop(['anomaly', 'changepoint'], axis=1)
                    pc = PCA(n_components=2).fit_transform(data)
                    df[['X', 'Y']] = pc
                    plt.figure()
                    sb.set(font='Times New Roman')
                    sns = sb.scatterplot(data=df, x='X', y='Y', hue='anomaly', palette='bright')
                    sns.set_title('The Anomaly Chart of  File {}'.format(ts))
                    sns.figure.savefig(path_to_plt + 'anom-{}.png'.format(ts))
                    plt.close('all')
                    print('The Chart of  File {} is Generated.'.format(ts))
                    ts += 1

    def IsolationForestExecutor(self):
        isofh = IsolationForestHandler(self.dc)
        isofh.findAnomalies(self.draw_chart, self.write_file)

    def KNNOExecutor(self):
        knno = KNNOHandler(self.dc)
        knno.findAnomalies(self.draw_chart, self.write_file)

    def LOFExecutor(self):
        lof = LOFHandler(self.dc)
        lof.findAnomalies(self.draw_chart, self.write_file)

    def RobustCovExecutor(self):
        rocov = RobustCovHandler(self.dc)
        rocov.findAnomalies(self.draw_chart, self.write_file)

    def OneClassSVMExecutor(self):
        ocsvm = OneClassSVMHandler(self.dc)
        ocsvm.findAnomalies(self.draw_chart, self.write_file)

    def SSDOExecutor(self):
        ssdo = SSDOHandler(self.dc)
        ssdo.findAnomalies(self.draw_chart, self.write_file)

    def SSkNNOExecutor(self):
        ssknno = SSkNNOHandler(self.dc)
        ssknno.findAnomalies(self.draw_chart, self.write_file)
