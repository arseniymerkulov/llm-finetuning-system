import requests
import logging


from settings import Settings
settings = Settings.get_instance()
logger = logging.getLogger(__name__)


class TestProcess:
    def __init__(self, name, configuration: dict):
        self.name = name
        self.logger = logging.getLogger(name)
        self.configuration = configuration

    def _create_data(self, attribute):
        return {
            'field': attribute,
            'value': self.configuration[attribute]
        }

    @staticmethod
    def start_run():
        response = requests.post(f'{settings.app_endpoint}/api/start')
        logger.info(response.text)
        assert 'error' not in response.json()

        return response

    def update_configuration(self, attribute):
        response = requests.post(f'{settings.app_endpoint}/api/update', json=self._create_data(attribute))
        logger.info(response.text)
        assert 'error' not in response.json()

        return response
