import selenium.common.exceptions as selenium_exceptions
from data_fetch.fetcher import Fetcher
from selenium import webdriver
from pandas import DataFrame
from threading import Thread
from numpy import ndarray
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
import os


class Crawler(Fetcher):
    def __init__(self, num_workers=10, overwrite=False):
        blacklist_path = "Data Storage\\Blacklists\\morningstar_blacklist.txt"
        super(Crawler, self).__init__(blacklist_path)
        self.existing_files = os.listdir("Data Storage\\Raw Stock Data")
        self.not_existing_stocks = []
        self.overwrite = overwrite
        self.num_workers = num_workers
        self.workers = []

    def __del__(self):
        self.blacklist = list(set(self.blacklist))
        self.save_blacklist()

    def load_data(self):
        all_stocks = pd.read_csv("Data Storage/NASDAQ_stocks.csv")['Symbol']
        for stock in all_stocks:
            if (f"{stock}.csv" not in self.existing_files or self.overwrite) and stock not in self.blacklist:
                self.not_existing_stocks.append(stock)
        stocks_chunks = np.array_split(self.not_existing_stocks, self.num_workers)
        return stocks_chunks

    def run(self):
        stocks_chunks = self.load_data()
        with tqdm(total=len(self.not_existing_stocks), desc="Getting Stock Data") as pbar:
            for i in range(self.num_workers):
                chunk = stocks_chunks[i]
                if len(chunk) > 1:
                    driver = DriverThread(chunk, pbar)
                    worker = Thread(target=driver.run_driver)
                    self.workers.append(worker)
                    worker.start()

            for worker in self.workers:
                if isinstance(worker, Thread):
                    worker.join()


class DriverThread(Crawler):
    def __init__(self, stocks: ndarray, pbar: tqdm):
        super(DriverThread, self).__init__()
        self.pbar = pbar
        self.save_path = "Data Storage\\Raw Stock Data\\"
        self.base_url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={}&type=10-K&dateb=&owner=exclude&count=40&search_text="
        self.current_stock = ""
        self.wait_iterations = 25
        self.wait_for_page_time = 5
        self.idle_wait_time = 0.05
        self.driver = webdriver.Firefox(executable_path="Web Driver\\geckodriver.exe")
        self.stocks = stocks

    def __del__(self):
        self.driver.close()

    def load_page(self) -> bool:
        for stock in self.stocks:
            if stock not in self.blacklist:
                try:
                    self.driver.get(self.base_url.format('AAME'))#stock))
                    self.current_stock = stock
                except selenium_exceptions.TimeoutException as e:
                    continue
                self.pbar.update(1)
                self.stocks = np.delete(self.stocks, 0)
                return True
        return False

    def get_all_stock_data(self):
        buttons = self.driver.find_elements_by_id("interactiveDataBtn")
        df_list = []
        for button in buttons:
            df_list.append(self.get_year_data(button))
        pd.concat(df_list).to_csv("test.csv")

    def get_year_data(self, button) -> DataFrame:
        year_data = []
        sections = self.get_year_sections(button)
        for section in sections:
            section.click()
            time.sleep(self.idle_wait_time)
            year_data.append(self.get_section_data())
        return pd.concat(year_data)

    def get_year_sections(self, button) -> list:
        self.driver.execute_script(f'''window.open("{button.get_attribute("href")}","_blank");''')
        self.driver.switch_to_window(self.driver.window_handles[-1])
        time.sleep(self.wait_for_page_time)
        while True:
            try:
                self.driver.find_element_by_id("menu_cat2").click()
                continue
            except selenium_exceptions.NoSuchElementException:
                time.sleep(self.idle_wait_time)
            break
        subsections = []

        # report 1 is useless
        for i in range(2, 1000):
            try:
                subsections.append(self.driver.find_element_by_id(f"r{i}"))
            except selenium_exceptions.NoSuchElementException:
                break
        return subsections

    def get_section_data(self) -> DataFrame:
        table_html = self.driver.find_element_by_class_name("report").get_attribute('outerHTML')
        df = pd.read_html(table_html)[0]
        df.set_index(df.columns[0], inplace=True)
        df = df.applymap(lambda x: str(x).strip("() $"))
        df = df.applymap(lambda x: x.replace(",", "."))
        return df

    def run_driver(self):
        while self.load_page():
            self.get_all_stock_data()
            if self.current_stock+".csv" not in os.listdir(self.save_path[:-1]):
                self.blacklist = self.read_blacklist()
                self.blacklist.append(self.current_stock)
                self.save_blacklist()
