import pandas as pd
import glob


from core.main_process.pipeline import Stage
from core.configuration.configuration import DatasetStorageFormat


class DataProcessing(Stage):
    def execute(self):
        # TODO: adapt for different data storage formats
        self.config.configure('dataset_storage_format', DatasetStorageFormat.CSV_TABLE)

        # lets generalize all common nlp datasets to 2 cases: datasets with
        # single text - single text correspondence for language modeling and
        # datasets with single text - integer correspondence for classification
        self.config.wait('dataset_table_columns')
        assert len(self.config.dataset_table_columns) == 2, 'more than 2 columns specified'

        self.config.wait('dataset_partition')
        self._load_dataset_from_csv_table()

    def _load_dataset_from_csv_table(self):
        # how to process several csv files? Is it possible use case?
        path = glob.glob(f'{self.config.dataset_dir}/*.csv')
        assert len(path), 'no .csv files in the dataset directory'

        df = pd.read_csv(path[0], encoding='latin-1')
        df = df[self.config.dataset_table_columns]
        df = df.dropna()

        # todo: balance partition for classification datasets
        if self.config.dataset_partition:
            df = df[:self.config.dataset_partition]

        self.config.configure('X', df[self.config.dataset_table_columns[0]].tolist())
        self.config.configure('Y', df[self.config.dataset_table_columns[1]].tolist())

    def validate(self) -> bool:
        pass
