import logging
import abc


from core.configuration.configuration import Configuration
from core.settings import Settings


class Pipeline:
    def __init__(self, *args):
        self.stages: list[Stage] = [*args]

    def execute(self):
        for stage in self.stages:
            stage.execute()

            # TODO: process validation result
            stage.validate()


class Stage(abc.ABC):
    def __init__(self):
        self.logger = logging.getLogger(Stage.__name__)
        self.config = Configuration.get_instance()
        self.settings = Settings.get_instance()

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def validate(self) -> bool:
        pass
