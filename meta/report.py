from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from typing import Literal
import logging
import getopt
import wandb
import math
import json
import csv
import sys

try:
    from core.settings import Settings
    from core.configuration.hyperparams import Task, FinetuningMethod
except ModuleNotFoundError:
    import pathlib

    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from core.settings import Settings
    from core.configuration.hyperparams import Task, FinetuningMethod


class Logger:
    settings = Settings.get_instance()
    wandb_token_path = 'data/wandb.json'
    storage_secrets_path = 'data/client_secrets.json'
    storage_auth_path = 'data/google_drive.auth'
    storage_dir = 'meta-datasets'
    report_filename = 'report'

    def __init__(self):
        self.logger = logging.getLogger('Logger')

        wandb.login(key=self._load_token(Logger.wandb_token_path))
        self.wandb_api = wandb.Api()

        GoogleAuth.DEFAULT_SETTINGS['client_config_file'] = Logger.storage_secrets_path
        storage_auth = GoogleAuth()

        storage_auth.LoadCredentialsFile(Logger.storage_auth_path)
        if storage_auth.credentials is None:
            storage_auth.LocalWebserverAuth()
        elif storage_auth.access_token_expired:
            storage_auth.Refresh()
        else:
            storage_auth.Authorize()

        storage_auth.SaveCredentialsFile(Logger.storage_auth_path)
        self.storage = GoogleDrive(storage_auth)

    def _get_cloud_dir_id(self):
        return self.storage.ListFile(
            {'q': f"title='{Logger.storage_dir}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}
        ).GetList()[0]['id']

    def _clear_cloud_report(self, cloud_filename, directory_id):
        reports = self.storage.ListFile(
            {'q': f"'{directory_id}' in parents and trashed=false"}
        ).GetList()

        for report in reports:
            if cloud_filename in report['title']:
                report = self.storage.CreateFile({'id': report['id']})
                report.Delete()

    def _create_cloud_report(self, local_filename, cloud_filename, directory_id):
        report = self.storage.CreateFile({'title': cloud_filename, 'parents': [{'id': directory_id}]})
        report.SetContentFile(local_filename)
        report.Upload()

    def _get_data(self, project):
        def _prepare(config, summary):
            config.pop('dataset_dir', None)
            config.pop('dataset_file', None)
            config.pop('dataset_storage_format', None)
            config.pop('model_auto_class', None)
            config.pop('lora_task', None)
            config.pop('num_workers', None)
            config.pop('project_dir', None)
            config.pop('wandb_token_path', None)
            config.pop('kaggle_token_path', None)
            config.pop('checkpoints_dir', None)
            config.pop('best_checkpoint_path', None)
            config.pop('examples', None)

            if config['finetuning_method'] != FinetuningMethod.LORA.name:
                config.pop('lora_r', None)
                config.pop('lora_alpha', None)
                config.pop('lora_dropout', None)

            config['metrics'] = {key: summary[key] for key in summary.keys() if key[0] != '_' and key != 'epoch'}
            return config

        return [_prepare(run.config, run.summary) for run in self.wandb_api.runs(project)]

    # todo: do i need to round values in dataset or do it for filtering only?
    def _filter_run(self, run, grouping_parameter):
        def adaptive_round(number: int, method: Literal['round', 'ceil'] = 'ceil', offset=0):
            digits = len(str(number)) - offset
            number = number / 10 ** digits
            number = round(number) if method == 'round' else math.ceil(number)
            return number * 10 ** digits

        run = run.copy()
        run.pop('metrics', None)
        run.pop('dataset_sample_min_length', None)
        run.pop('dataset_sample_max_length', None)
        run.pop('dataset_sample_avg_length', None)
        run.pop(grouping_parameter, None)

        run['categories'] = tuple(run['categories'])
        run['dataset_balance'] = tuple(run['dataset_balance'].items())
        return run

    def _filter_data(self, runs, grouping_parameter):
        runs_filtered = [self._filter_run(run, grouping_parameter) for run in runs]
        runs_filtered_dict = {frozenset(item.items()): item for item in runs_filtered}
        runs_filtered = list(runs_filtered_dict.values())

        for run in runs_filtered:
            self.logger.info(f'{run["epochs"]},'
                             f'{run["dataset_balance"]},'
                             f'{run["dataset_instances"]},')
        self.logger.info(f'found {len(runs_filtered)} group(s) of experiments')

        return runs_filtered, runs_filtered_dict

    def _select_data(self, runs, grouping_parameter):
        runs_filtered, runs_filtered_dict = self._filter_data(runs, grouping_parameter)
        runs_filtered_dict = {key: [] for key in runs_filtered_dict}
        selected = []

        for run in runs:
            key = frozenset(self._filter_run(run, grouping_parameter).items())
            runs_filtered_dict[key].append(run)

        for key in runs_filtered_dict:
            best = None
            similar_runs = runs_filtered_dict[key]
            task = Task[similar_runs[0]['task']]

            if task == Task.CLASSIFICATION:
                similar_runs = sorted(similar_runs, key=lambda run: run['metrics']['Accuracy'], reverse=True)
                best = similar_runs[0]
                best['selected_from'] = [run['metrics'] for run in similar_runs]

            selected.append(best)
        return selected

    def upload_report(self, project):
        runs = self._get_data(project)
        runs = self._select_data(runs, 'additional_learning_rate')

        Logger.save_json(f'{Logger.settings.projects_dir}/{project}/{Logger.report_filename}.json', runs)
        # Logger.save_csv(f'{Logger.reports_dir}/{project}.csv', runs)

        directory_id = self._get_cloud_dir_id()
        self._clear_cloud_report(project, directory_id)
        self._create_cloud_report(f'{Logger.settings.projects_dir}/{project}/{Logger.report_filename}.json',
                                  f'{project}.json',
                                  directory_id)
        self.logger.info('report uploaded to storage')

    @staticmethod
    def save_json(path, runs):
        with open(path, 'w') as file:
            for run in runs:
                file.write(json.dumps(run))
                file.write('\n')

    @staticmethod
    def save_csv(path, runs):
        with open(path, 'w') as file:
            assert len(runs), 'empty runs results'
            config, _ = runs[0]
            keys = list(config.keys())

            if 'metrics' not in keys:
                keys.append('metrics')

            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()

            for config, summary in runs:
                if 'metrics' not in config:
                    config['metrics'] = Logger._get_metrics(summary)
                writer.writerow(config)

    @staticmethod
    def _load_token(token_path):
        with open(token_path, 'r') as file:
            return json.loads(file.read())['key']

    @staticmethod
    def _get_metrics(summary: dict):
        # todo: logic for metric extraction can be improved
        return {key: summary[key] for key in summary.keys() if key[0] != '_' and key != 'epoch'}


def main():
    arguments = sys.argv[1:]
    arguments, _ = getopt.getopt(arguments, 'p:', 'project=')
    assert len(arguments), 'you need to specify project with "--project" parameter'
    project = arguments[0][1]

    logging.basicConfig(level=logging.INFO)
    Logger().upload_report(project)


if __name__ == '__main__':
    main()
