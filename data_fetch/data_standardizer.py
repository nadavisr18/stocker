from data_fetch.fetcher import Fetcher
from matplotlib import pyplot as plt
from multiprocessing import Pool
from pandas import DataFrame
from numpy import array
from tqdm import tqdm
import pandas as pd
import uuid
import os


class Standardizer(Fetcher):
    def __init__(self, num_workers=10, overwrite=False, null_threshold=8, years_per_sample=6):
        blacklist_path = "Data Storage\\Blacklists\\standardizer_blacklist.txt"
        super(Standardizer, self).__init__(blacklist_path)
        self.save_path = "Data Storage\\Processed Stock Data\\"
        self.stocks_data_path = "Data Storage\\Full Stock Data\\"
        self.existing_files = os.listdir(self.save_path)
        self.years_per_sample = years_per_sample
        self.null_threshold = null_threshold
        self.num_workers = num_workers
        self.overwrite = overwrite
        self.stocks = self.load_data()

    def load_data(self) -> list:
        stocks = []
        for file in tqdm(os.listdir(self.stocks_data_path), desc="Loading Full Stock Data"):
            if (file not in self.existing_files and file.strip(".csv") not in self.blacklist) or self.overwrite:
                stock_data = pd.read_csv(self.stocks_data_path+file, index_col=0)
                stock_data = stock_data[stock_data.index != 'TTM']
                stocks.append({file.strip(".csv"): stock_data})
        return stocks

    def run(self):
        with Pool(self.num_workers) as pool:
            for _ in tqdm(pool.imap_unordered(self.stock_training_data, self.stocks), total=len(self.stocks), desc="Generating Training Data"):
                pass

    def check_null_amount(self):
        row_nulls = []
        for stock in tqdm(self.stocks, desc="Generating Histogram"):
            stock_df = list(stock.values())[0]
            row_nulls.extend(self.count_nulls(self.normalize_df(stock_df)))
        plt.hist(row_nulls,bins=30,alpha=0.8)
        plt.xlabel("Null Occurrences per Row")
        plt.ylabel("Number of Rows")
        plt.title("Null Values")
        plt.show()

    def stock_training_data(self, stock: dict):
        stock_name, stock_df = tuple(stock.items())[0]
        training_data = []
        for i in range(len(stock_df)-self.years_per_sample):
            sample = stock_df[i:i+self.years_per_sample]
            relevant_data = self.normalize_df(sample)
            if not any(self.count_nulls(relevant_data) > self.null_threshold):
                relevant_data = relevant_data.stack().reset_index(drop=True)
                relevant_data.name = sample.iloc[-1].name
                training_data.append(relevant_data.to_frame().transpose())
        if len(training_data) > 0:
            final_data = pd.concat(training_data)
            final_data.to_csv(self.save_path+stock_name+".csv")
        else:
            self.blacklist = self.read_blacklist()
            self.blacklist.append(stock_name)
            self.save_blacklist()

    def count_samples(self, null_threshold: int) -> list:
        count = 0
        for stock in self.stocks:
            stock_name, stock_df = tuple(stock.items())[0]
            for i in range(len(stock_df)-self.years_per_sample):
                sample = stock_df[i:i+self.years_per_sample]
                relevant_data = self.normalize_df(sample)
                if not any(self.count_nulls(relevant_data) > null_threshold):
                    count += 1
        return [null_threshold, count]

    def create_threshold_samples_graph(self):
        threshold_to_samples = []
        thresholds = list(range(1, 30))
        with Pool(self.num_workers) as pool:
            for result in tqdm(pool.imap_unordered(self.count_samples, thresholds), total=len(thresholds), desc="Generating Graph Data"):
                threshold_to_samples.append(result)
        x, y = zip(*threshold_to_samples)
        plt.scatter(x, y)
        plt.xlabel("Threshold")
        plt.ylabel("Number of Samples")
        plt.title("Null Values Threshold to Samples")
        plt.show()

    @staticmethod
    def normalize_df(df: DataFrame) -> DataFrame:
        cleaned_df = df.applymap(lambda x: float(x) if not x == '—' else None)
        normalized_df = cleaned_df.apply(lambda x: (x - x.mean()) / x.std())
        normalized_df.fillna(0, inplace=True)
        return normalized_df

    @staticmethod
    def count_nulls(df: DataFrame) -> array:
        nulls = df.apply(lambda x: x.value_counts()[0] if 0 in x.values else 0, axis=1).values.tolist()
        return array(nulls)
