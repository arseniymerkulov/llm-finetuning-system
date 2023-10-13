import logging
import abc


from core.configuration import Configuration
from core.settings import Settings


class Pipeline:
    def __init__(self, *args):
        # todo: check that loggers have readable names in log (they don't)
        self.logger = logging.getLogger(Stage.__name__)
        self.config = Configuration.get_instance()
        self.stages: list[Stage] = [*args]

    def execute(self):
        self.logger.info(f'starting pipeline: {[type(stage).__name__ for stage in self.stages]}')

        for stage in self.stages:
            try:
                self.logger.info(f'starting new stage: {type(stage).__name__}')
                self.config.configure_status('stage', type(stage).__name__)
                self.config.configure_status('phase', 'execute')
                self.logger.info(f'executing {self.config.status["stage"]}:{self.config.status["phase"]}')
                stage.execute()

                self.config.configure_status('phase', 'validate')
                self.logger.info(f'executing {self.config.status["stage"]}:{self.config.status["phase"]}')
                # TODO: process validation result
                stage.validate()

                self.config.configure_status('phase', 'approve')
                self.config.wait_for_approval()
                self.config.configure_status('phase', 'finished')
                self.logger.info(f'stage {self.config.status["stage"]} is finished')

            except AssertionError as e:
                self.config.configure_status('error', str(e))
                self.logger.error(str(e))
                return
        return


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
