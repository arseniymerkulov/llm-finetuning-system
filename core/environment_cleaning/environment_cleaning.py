import wandb


from core.configuration import Configuration
from core.main_process.pipeline import Stage


# todo: think about fusing this package with environment_setup
class EnvironmentCleaning(Stage):
    def execute(self):
        wandb.finish()
        Configuration.reset()
        self.config.configure_status('stage', EnvironmentCleaning.__name__)
        self.config.configure_status('phase', 'execute')

    def validate(self) -> bool:
        pass
