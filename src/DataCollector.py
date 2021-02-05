import os
import numpy as np
import pandas as pd
from colorama import Fore


class DataCollector:
    path = '../datases/SKAB/'
    data = []
    csv_files = []
    log = False

    def __init__(self, log=False):
        self.log = log
        if log:
            print(Fore.GREEN + "\t+----------------------------------------------")
            print(Fore.GREEN + "\tRead HAR Dataset Begin.")
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".csv"):
                    self.csv_files.append(os.path.join(root, file))
                    if log:
                        print(Fore.GREEN + "\tFind {} in {}.".format(file, root))

        if log:
            print(Fore.BLUE + "\tFind {} CSV Dataset Files.".format(len(self.csv_files)))
            print(Fore.GREEN + "\t+----------------------------------------------")

    def dataSetReader(self):

        if not os.path.isfile(self.path) and len(self.csv_files) <= 0:
            print(Fore.RED + "\t+------------------------+")
            print(Fore.RED + "\t| File or Path Incorrect |")
            print(Fore.RED + "\t+------------------------+")

        dfs = []
        for path in self.csv_files:
            df = pd.read_csv(path, index_col='datetime', sep=';', parse_dates=True)
            dfs.append(df)
        if self.log:
            print(Fore.GREEN + "\t+----------------------------------------------")
            print('\tFeatures:')
            for col in dfs[2].columns:
                print('\t\t• ', col)
            print(Fore.GREEN + "\t+----------------------------------------------")
        self.data = dfs
        return dfs

    def getData(self):
        return self.data

    def getAnomalyFree(self):
        anomaly_free_df = pd.read_csv([file for file in self.csv_files if 'anomaly-free' in file][0],
                                      sep=';', index_col='datetime', parse_dates=True)
        return anomaly_free_df

    def getWithAnomaly(self):
        list_of_df = [pd.read_csv(file, sep=';', index_col='datetime',
                                  parse_dates=True) for file in self.csv_files if 'anomaly-free' not in file]
        return list_of_df
