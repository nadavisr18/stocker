from multiprocessing import Pool
from itertools import islice
from pandas import Series
from tqdm import tqdm
import pandas as pd
import numpy as np
import yfinance
import re
import os


class YahooFetcher:
    def __init__(self, num_workers=10, overwrite=False):
        self.stocks_data_path = "Data Storage\\Raw Stock Data\\"
        self.save_path = "Data Storage\\Full Stock Data\\"
        self.existing_files = os.listdir(self.save_path)
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.stocks = self.load_data()

    def load_data(self) -> list:
        stocks = []
        for file in tqdm(os.listdir(self.stocks_data_path), desc="Loading Raw Stock Data"):
            if file not in self.existing_files or self.overwrite:
                stock_data = pd.read_csv(self.stocks_data_path+file, index_col=0)
                stock_data = stock_data[stock_data.index != 'TTM']
                stocks.append({file.strip(".csv"): stock_data})
        return stocks

    def run(self):
        with Pool(self.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self.add_yahoo_data, self.stocks), total=len(self.stocks), desc="Processing Raw Data"):
                pass

    def add_yahoo_data(self, stocks):
        for symbol, stock_data in stocks.items():
            yahoo_data = stock_data.apply(lambda x: self.get_stock_data(symbol, x.name), axis=1)
            enhanced_stock_data = stock_data.merge(yahoo_data, left_index=True, right_index=True)
            enhanced_stock_data.to_csv(self.save_path+symbol+".csv")

    @staticmethod
    def get_stock_data(symbol: str, date: str) -> Series:
        year = re.findall(r'\d{4}', date)[0]
        start_date = year+"-01-01"
        end_date = year+"-12-31"
        share = yfinance.Ticker(symbol)
        history_list = share.history(start=start_date, end=end_date,)
        history = pd.DataFrame(history_list)
        summarized_data = history.describe()
        summarized_data = summarized_data[summarized_data.index != 'count']
        relevant_data = summarized_data[['Open', 'Volume']]
        relevant_data['Volume'] = relevant_data['Volume'].apply(lambda x: np.log10(x))
        relevant_data = relevant_data.stack()
        relevant_data.index = relevant_data.index.map('{0[1]}/{0[0]}'.format)
        return relevant_data

    @staticmethod
    def chunks(data, chunks):
        size = len(data)//chunks
        it = iter(data)
        ret_chunks = []
        for i in range(0, len(data), size):
            ret_chunks.append({k: data[k] for k in islice(it, size)})
        return ret_chunks
