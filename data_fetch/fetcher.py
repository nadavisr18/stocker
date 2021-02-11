import time


class Fetcher:
    def __init__(self, path):
        self.blacklist_path = path
        self.blacklist = self.read_blacklist()

    def read_blacklist(self):
        while True:
            try:
                file = open(self.blacklist_path, 'r')
                bl = file.readlines()
                bl = [stock.strip("\n") for stock in bl]
                file.close()
            except PermissionError as e:
                time.sleep(0.02)
                continue
            return bl

    def save_blacklist(self):
        while True:
            try:
                file = open(self.blacklist_path, 'w')
                for s in self.blacklist:
                    file.write(s + "\n")
                file.close()
            except PermissionError as e:
                time.sleep(0.02)
                continue
            return
