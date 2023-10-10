import pandas as pd
import glob


from core.main_process.pipeline import Stage
from core.configuration import DatasetStorageFormat, Task


class DataProcessing(Stage):
    def execute(self):
        # todo: support 'text files in category folders' format
        # todo: field is useless
        self.config.configure('dataset_storage_format', DatasetStorageFormat.TABLE)
        self.config.wait('dataset_table_columns')
        assert len(self.config.dataset_table_columns) == 2, 'more than 2 columns specified'

        self.config.wait('dataset_partition')
        if self.config.task == Task.CLASSIFICATION:
            self.config.wait('dataset_balance')

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

        if self.config.task == Task.CLASSIFICATION:
            num_classes, categories = DataProcessing._preprocess_categories(
                df[self.config.dataset_table_columns[1]].tolist()
            )
            self.config.configure('num_classes', num_classes)
            self.config.configure('categories', categories)

            if self.config.dataset_balance:
                df = self._balance_dataset(df)

            self.logger.info(df[self.config.dataset_table_columns[1]].value_counts())
            self.logger.info(df.head())

            self.config.configure('X', df[self.config.dataset_table_columns[0]].tolist())
            self.config.configure('Y', self._process_classification_labels(
                df[self.config.dataset_table_columns[1]].tolist()
            ))

        else:
            if self.config.dataset_partition:
                df = df[:self.config.dataset_partition]

            self.logger.info(df.head())

            self.config.configure('X', df[self.config.dataset_table_columns[0]].tolist())
            self.config.configure('Y', df[self.config.dataset_table_columns[1]].tolist())
            assert isinstance(self.config.Y[0], str), \
                f'only <str> type is available for targets with "{self.config.task.name}" task'

        assert len(self.config.X) and len(self.config.Y), 'empty dataset'
        assert isinstance(self.config.X[0], str), 'X is not <str> type'

    def _balance_dataset(self, df):
        df = df.groupby(self.config.dataset_table_columns[1])
        sample_size = min(df.size().min(), self.config.dataset_partition // self.config.num_classes) \
            if self.config.dataset_partition else df.size().min()
        return pd.DataFrame(df.apply(lambda x: x.sample(sample_size).reset_index(drop=True)))

    @staticmethod
    def _preprocess_categories(Y):
        if isinstance(Y[0], bool):
            return 2, ['False', 'True']

        elif isinstance(Y[0], int):
            num_classes = max(set(Y)) + 1
            return num_classes, [str(i) for i in range(num_classes)]

        if isinstance(Y[0], str):
            categories = list(set(Y))
            return len(categories), categories

    def _process_classification_labels(self, Y):
        if isinstance(Y[0], str):
            return [self.config.categories.index(label) for label in Y]

        return Y

    def validate(self) -> bool:
        pass
