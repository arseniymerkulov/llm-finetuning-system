from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import logging
import getopt
import wandb
import json
import csv
import sys
import os


class Logger:
    # todo: add report to project dir
    wandb_token_path = 'data/wandb.json'
    storage_secrets_path = 'data/client_secrets.json'
    storage_auth_path = 'data/google_drive.auth'
    storage_dir = 'meta-datasets'
    reports_dir = 'data/reports'

    def __init__(self):
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

    def upload_report(self, project):
        runs = [(run.config, run.summary) for run in self.wandb_api.runs(project)]

        if not os.path.exists(Logger.reports_dir):
            os.mkdir(Logger.reports_dir)

        Logger.save_json(f'{Logger.reports_dir}/{project}.json', runs)
        Logger.save_csv(f'{Logger.reports_dir}/{project}.csv', runs)

        directory_id = self.storage.ListFile(
            {'q': f"title='{Logger.storage_dir}' and mimeType='application/vnd.google-apps.folder' and trashed=false"}
        ).GetList()[0]['id']

        reports = self.storage.ListFile(
            {'q': f"'{directory_id}' in parents and trashed=false"}
        ).GetList()

        for report in reports:
            if report['title'] == f'{project}.json' or report['title'] == f'{project}.csv':
                report = self.storage.CreateFile({'id': report['id']})
                report.Delete()

        report = self.storage.CreateFile({'title': f'{project}.json', 'parents': [{'id': directory_id}]})
        report.SetContentFile(f'{Logger.reports_dir}/{project}.json')
        report.Upload()

        report = self.storage.CreateFile({'title': f'{project}.csv', 'parents': [{'id': directory_id}]})
        report.SetContentFile(f'{Logger.reports_dir}/{project}.csv')
        report.Upload()

    @staticmethod
    def save_json(path, runs):
        with open(path, 'w') as file:
            for config, summary in runs:
                if 'metrics' not in config:
                    config['metrics'] = Logger._get_metrics(summary)

                file.write(json.dumps(config))
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

    Logger().upload_report(project)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info(f'report uploaded to storage')


if __name__ == '__main__':
    main()
