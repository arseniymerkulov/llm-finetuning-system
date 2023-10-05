import logging


class Settings:
    logging.basicConfig(level=logging.INFO)
    _instance = None

    def __init__(self):
        self.app_endpoint = 'http://127.0.0.1:5000'

    @staticmethod
    def get_instance():
        if not Settings._instance:
            Settings._instance = Settings()

        return Settings._instance
