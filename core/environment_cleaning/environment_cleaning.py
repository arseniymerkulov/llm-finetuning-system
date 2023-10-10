import wandb


from core.configuration import Configuration
from core.main_process.pipeline import Stage


class EnvironmentCleaning(Stage):
    def execute(self):
        wandb.finish()
        Configuration.reset()

    def validate(self) -> bool:
        pass
