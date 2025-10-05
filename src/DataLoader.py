import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, path):
        self._path = path
    def loadDataOneCsv(self, stockticker):
        ticker = stockticker.upper()
        wholepath = f"{self._path}/{ticker}.csv"
        data = pd.read_csv(wholepath)
        return data

    def loadAllDataCsv(self):
        dataset = {}
        with open(self._path + "/tickers.txt") as csvfile:
            for line in csvfile:
                dataset[line.strip()] = self.loadDataOneCsv(line.strip())

        return dataset