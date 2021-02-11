from data_fetch.web_crawler import Crawler
from data_fetch.yahoo_fetcher import YahooFetcher
from data_fetch.data_standardizer import Standardizer
from model.data_prep import DataPerp
from model.model import MLModel
import yaml


def main():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    crawler = Crawler(config['Crawler']['num_workers'],
                      config['Crawler']['overwrite'])
    crawler.run()

    yahoo = YahooFetcher(config['YahooFetcher']['num_workers'],
                         config['YahooFetcher']['overwrite'])
    yahoo.run()

    stand = Standardizer(config['Standardizer']['num_workers'],
                         config['Standardizer']['overwrite'],
                         config['Standardizer']['null_threshold'],
                         config['Standardizer']['years_per_sample'])
    stand.run()

    prep = DataPerp(config['DataPrep']['num_workers'],
                    config['DataPrep']['overwrite'],
                    config['DataPrep']['years'],
                    config['DataPrep']['base_symbol'])
    prep.prepare_data()

    model = MLModel(config['Model']['type'],
                    config['Model']['test_train_ratio'],
                    config['RandomForest']['trees'],
                    config['NeuralNetwork']['layer_sizes'],
                    config['NeuralNetwork']['activation'],
                    config['NeuralNetwork']['solver'],
                    config['NeuralNetwork']['batch_size'],
                    config['NeuralNetwork']['learning_rate'])
    model.train_model()



if __name__ == '__main__':
    main()