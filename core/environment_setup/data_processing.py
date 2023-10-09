import pandas as pd
import numpy as np
import glob


from core.main_process.pipeline import Stage
from core.configuration import DatasetStorageFormat, Task


class DataProcessing(Stage):
    def execute(self):
        # TODO: adapt for different data storage formats
        # todo: support 'text files in category folders' format
        # todo: field is useless
        self.config.configure('dataset_storage_format', DatasetStorageFormat.TABLE)

        # lets generalize all common nlp datasets to 2 cases: datasets with
        # single text - single text correspondence for language modeling and
        # datasets with single text - integer correspondence for classification
        # todo: wait only for tables
        self.config.wait('dataset_table_columns')
        assert len(self.config.dataset_table_columns) == 2, 'more than 2 columns specified'

        self.config.wait('dataset_partition')
        self._load_dataset_with_pandas()

    def _load_dataset_with_pandas(self):
        path = glob.glob(f'{self.config.dataset_dir}/*.csv') + \
               glob.glob(f'{self.config.dataset_dir}/*.json')
        assert len(path), 'no .csv/.json files in the dataset directory'
        dataset_path = path[0]

        if len(path) > 1:
            self.config.wait('dataset_file')
            dataset_path = f'{self.config.dataset_dir}/{self.config.dataset_file}'

        if 'csv' in dataset_path:
            # todo: how to estimate encoding?
            #       different encodings with try:
            df = pd.read_csv(dataset_path, encoding='latin-1')
        else:
            df = pd.read_json(dataset_path, lines=True)

        df = df[self.config.dataset_table_columns]
        df = df.dropna()

        # todo: balance partition for classification datasets
        if self.config.dataset_partition:
            df = df[:self.config.dataset_partition]

        self.config.configure('X', df[self.config.dataset_table_columns[0]].tolist())
        self.config.configure('Y', df[self.config.dataset_table_columns[1]].tolist())

        assert len(self.config.X), 'empty dataset'
        assert isinstance(self.config.X[0], str), 'X is not <str> type'

        if self.config.task == Task.CLASSIFICATION:
            self._process_classification_labels()
        else:
            assert isinstance(self.config.Y[0], str), \
                f'only <str> type is available for targets with "{self.config.task.name}" task'

    def _process_classification_labels(self):
        if isinstance(self.config.Y[0], bool):
            vector_labels = [[0, 1] if label else [1, 0] for label in self.config.Y]
            self.config.configure('Y', vector_labels)
            self.config.configure('num_classes', 2)
            self.config.configure('categories', ['False', 'True'])

        elif isinstance(self.config.Y[0], int):
            num_classes = max(set(self.config.Y)) + 1

            vector_labels = []
            for numeric_label in self.config.Y:
                label = list(np.zeros(num_classes, dtype=int))
                label[numeric_label] = 1
                vector_labels.append(label)

            self.config.configure('Y', vector_labels)
            self.config.configure('num_classes', num_classes)
            self.config.configure('categories', [str(i) for i in range(num_classes)])

        elif isinstance(self.config.Y[0], str):
            categories = list(set(self.config.Y))
            num_classes = len(categories)

            vector_labels = []
            for category in self.config.Y:
                label = list(np.zeros(num_classes, dtype=int))
                label[categories.index(category)] = 1
                vector_labels.append(label)

            self.config.configure('Y', vector_labels)
            self.config.configure('num_classes', num_classes)
            self.config.configure('categories', categories)

    def validate(self) -> bool:
        pass
