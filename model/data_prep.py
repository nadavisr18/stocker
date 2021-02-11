from dateutil import parser as parser
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import yfinance
import os


class DataPerp:
    def __init__(self, num_workers=10, overwrite=False, years=1, base_symbol="NDAQ"):
        self.stocks_data_path = "Data Storage\\Processed Stock Data\\"
        self.save_path = "Data Storage\\ML Ready Data\\"
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.years = years
        self.base_symbol = base_symbol

    def prepare_data(self):
        if not self.overwrite and "input.csv" in os.listdir(self.save_path) and "labels.csv" in os.listdir(self.save_path):
            return
        input_data = []
        labels = []
        stocks = os.listdir(self.stocks_data_path)
        with Pool(self.num_workers) as pool:
            for data, label in tqdm(pool.imap_unordered(self.prepare_stock_data, stocks), total=len(stocks), desc="Preparing Training Data"):
                if len(data) > 0 and len(label) > 0:
                    input_data.append(data)
                    labels.extend(label)
        pd.concat(input_data).to_csv(self.save_path+"input.csv", index=False)
        pd.Series(labels).to_csv(self.save_path+"labels.csv", index=False)

    def prepare_stock_data(self, stock) -> tuple:
        stock_data = pd.read_csv(self.stocks_data_path + stock, index_col=0)
        labels = []
        for index_date in stock_data.index:
            date = parser.parse(index_date)
            try:
                label = self.did_stock_outperform_market(symbol=stock.strip(".csv"), from_date=date)
                labels.append(label)
            except IndexError:
                stock_data = stock_data.drop(index=index_date)
        return stock_data, labels

    def did_stock_outperform_market(self, symbol, from_date) -> bool:
        to_date = from_date.replace(year=from_date.year+self.years)
        base_gain = self.get_gain(self.base_symbol, from_date, to_date)
        stock_gain = self.get_gain(symbol, from_date, to_date)
        return stock_gain > base_gain

    @staticmethod
    def get_gain(symbol, from_date, to_date) -> float:
        ticker = yfinance.Ticker(symbol)
        start_of_year = ticker.history(start=from_date.replace(day=from_date.day-2).strftime("%Y-%m-%d"),
                                       end=from_date.replace(day=from_date.day+2).strftime("%Y-%m-%d")).mean()['Open']
        end_of_year = ticker.history(start=to_date.replace(day=to_date.day-2).strftime("%Y-%m-%d"),
                                     end=to_date.replace(day=to_date.day+2).strftime("%Y-%m-%d")).mean()['Open']
        gain = (end_of_year - start_of_year) / start_of_year
        return gain
