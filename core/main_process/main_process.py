from threading import Thread


from core.main_process.pipeline import Pipeline
from core.configuration import PipelineSetup
from core.environment_setup import (
    EnvironmentSetup,
    DataCollecting,
    DataProcessing,
    ModelSelection,
    DataTokenizing
)
from core.finetuning import (HPO, FinetuningMethodSelection, Finetuning)
from core.evaluation import Evaluation
from core.environment_cleaning import EnvironmentCleaning


class MainProcess(Thread):
    def __init__(self, pipeline_setup: PipelineSetup):
        super().__init__()

        stages = [
            EnvironmentCleaning(),
            EnvironmentSetup(),
            DataCollecting(),
            DataProcessing(),
            ModelSelection(),
            DataTokenizing(),
            FinetuningMethodSelection(),
            HPO()
        ]

        if pipeline_setup == PipelineSetup.FULL:
            stages.append(Finetuning())
            stages.append(Evaluation())

        elif pipeline_setup == PipelineSetup.EVALUATION:
            stages.append(Evaluation())

        stages.append(EnvironmentCleaning())
        self.pipeline = Pipeline(*stages)

    def run(self):
        self.pipeline.execute()
