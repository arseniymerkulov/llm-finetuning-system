from threading import Thread


from core.main_process.pipeline import Pipeline
from core.environment_setup import *
from core.finetuning import *
from core.evaluating import *


class MainProcess(Thread):
    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline(
            EnvironmentSetup(),
            DataCollecting(),
            DataProcessing(),
            ModelSelection(),
            DataTokenizing(),
            FinetuningMethodSelection(),
            HPO(),
            # Finetuning(),
            Evaluating()
        )

    def run(self):
        self.pipeline.execute()
