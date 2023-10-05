import json
import os


from core.main_process.pipeline import Stage


class DataCollecting(Stage):
    def execute(self):
        # dataset load type: enum = ...
        # load from kaggle, transformers
        self.config.configure('kaggle_token_path', 'data/kaggle.json')

        self.config.wait('dataset_alias')
        self.config.configure('dataset_dir', f'{self.config.project_dir}/'
                                             f'{self.settings.datasets_dir}/'
                                             f'{self.config.dataset_alias}')

        self._load_kaggle_token()
        self._load_from_kaggle()

    def validate(self) -> bool:
        pass

    def _load_kaggle_token(self):
        with open(self.config.kaggle_token_path, 'r') as file:
            data = json.loads(file.read())

        os.environ['KAGGLE_USERNAME'] = data['username']
        os.environ['KAGGLE_KEY'] = data['key']

    def _load_from_kaggle(self):
        import kaggle
        
        if not os.path.exists(self.config.dataset_dir):
            os.makedirs(self.config.dataset_dir, exist_ok=True)

            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(self.config.dataset_alias, path=self.config.dataset_dir, unzip=True)
