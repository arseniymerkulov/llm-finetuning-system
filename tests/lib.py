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

    def _create_data(self, attribute, forced=False):
        return {
            'field': attribute,
            'value': self.configuration[attribute],
            'forced': forced
        }

    def _request(self, method: Literal['get', 'post'], route, json=None):
        response = getattr(requests, method)(f'{settings.app_endpoint}/api/{route}', json=json)
        self.logger.info(response.text)
        assert 'error' not in response.json() or response.json()['error'] is None, response.json()['error']

        return response

    def start_run(self):
        response = self._request('post', 'start', {'pipeline_setup': self.configuration['pipeline_setup']})

        self.set_stage('EnvironmentCleaning')
        self.wait('approve')
        self.approve_stage()
        return response

    def update_configuration(self, attribute, forced=False):
        return self._request('post', 'update', json=self._create_data(attribute, forced))

    def set_stage(self, stage: Literal['EnvironmentCleaning',
                                       'EnvironmentSetup',
                                       'DataCollecting',
                                       'DataProcessing',
                                       'ModelSelection',
                                       'DataTokenizing',
                                       'HPO',
                                       'FinetuningMethodSelection',
                                       'Finetuning',
                                       'Evaluation']):
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

    def execute_pipeline(self):
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

        if 'dataset_need_balance' in self.configuration:
            self.update_configuration('dataset_need_balance')

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

            self.set_stage('Evaluation')
            self.wait('approve')
            self.approve_stage()

        if self.configuration['pipeline_setup'] == 'EVALUATION':
            self.set_stage('Evaluation')
            self.wait('approve')
            self.approve_stage()

        self.set_stage('EnvironmentCleaning')
        self.wait('approve')
        self.approve_stage()
        self.wait('finished')

    def selection(self, scenario: dict):
        def run(**kwargs):
            self.logger.info(f'start experiment with {kwargs}')
            self.start_run()
            for kwarg in kwargs:
                self.configuration[kwarg] = kwargs[kwarg]
                self.update_configuration(kwarg, forced=True)
            self.execute_pipeline()

        def rec(scenario: dict, **kwargs):
            if len(scenario.keys()) == 1:
                key = list(scenario.keys())[0]

                for value in scenario[key]:
                    kwargs[key] = value
                    run(**kwargs)
                return

            for key in scenario:
                new_scenario = scenario.copy()
                del new_scenario[key]

                for value in scenario[key]:
                    kwargs[key] = value
                    rec(new_scenario, **kwargs)
                return

        rec(scenario)
