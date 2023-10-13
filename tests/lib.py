from typing import Literal
import requests
import logging
import time


from core.settings import Settings
settings = Settings.get_instance()


class TestProcess:
    def __init__(self, name, configuration: dict):
        self.name = name
        self.logger = logging.getLogger(name)
        self.configuration = configuration
        self.stage = ''

    def _create_data(self, attribute):
        return {
            'field': attribute,
            'value': self.configuration[attribute]
        }

    def _request(self, method: Literal['get', 'post'], route, json=None):
        response = getattr(requests, method)(f'{settings.app_endpoint}/api/{route}', json=json)
        self.logger.info(response.text)
        assert 'error' not in response.json() or response.json()['error'] is None, response.json()['error']

        return response

    def start_run(self):
        return self._request('post', 'start', {'pipeline_setup': self.configuration['pipeline_setup']})

    def update_configuration(self, attribute):
        return self._request('post', 'update', json=self._create_data(attribute))

    def set_stage(self, stage: Literal['EnvironmentCleaning',
                                       'EnvironmentSetup',
                                       'DataCollecting',
                                       'DataProcessing',
                                       'ModelSelection',
                                       'DataTokenizing',
                                       'HPO',
                                       'FinetuningMethodSelection',
                                       'Finetuning',
                                       'Evaluating']):
        self.stage = stage

    def approve_stage(self):
        status = self.get_status().json()
        assert status['stage'] == self.stage and status['phase'] == 'approve'

        return self._request('post', 'approve')

    def get_status(self):
        return self._request('get', 'status')

    def wait(self, phase: Literal['execute', 'validate', 'approve', 'finished'] = 'approve'):
        status = self.get_status().json()
        while not (status['stage'] == self.stage and status['phase'] == phase):
            status = self.get_status().json()
            time.sleep(0.5)
        self.logger.info(f'stage changed phase, current status: {status}')

    def run_pipeline(self):
        self.start_run()

        self.set_stage('EnvironmentCleaning')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('EnvironmentSetup')
        self.wait('execute')
        self.update_configuration('project')
        self.update_configuration('pipeline_setup')
        self.update_configuration('task')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('DataCollecting')
        self.wait('execute')
        self.update_configuration('dataset_alias')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('DataProcessing')
        self.wait('execute')

        if 'dataset_file' in self.configuration:
            self.update_configuration('dataset_file')

        self.update_configuration('dataset_table_columns')

        if 'dataset_partition' in self.configuration:
            self.update_configuration('dataset_partition')

        if 'dataset_balance' in self.configuration:
            self.update_configuration('dataset_balance')

        self.wait('approve')
        self.approve_stage()

        self.set_stage('ModelSelection')
        self.wait('execute')
        self.update_configuration('model_alias')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('DataTokenizing')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('FinetuningMethodSelection')
        self.wait('approve')
        self.approve_stage()

        self.set_stage('HPO')
        self.wait('approve')
        self.approve_stage()

        if self.configuration['pipeline_setup'] == 'FULL':
            self.set_stage('Finetuning')
            self.wait('approve')
            self.approve_stage()

            # todo: rename stage to Evaluation
            self.set_stage('Evaluating')
            self.wait('approve')
            self.approve_stage()

        if self.configuration['pipeline_setup'] == 'EVALUATION':
            self.set_stage('Evaluating')
            self.wait('approve')
            self.approve_stage()

        self.set_stage('EnvironmentCleaning')
        self.wait('approve')
        self.approve_stage()
        self.wait('finished')
