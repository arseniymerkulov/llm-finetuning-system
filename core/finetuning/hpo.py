from core.main_process.pipeline import Stage
from core.configuration import (
    TASK_TO_LORA_TASK_MAPPING,
    ModelOptimizer,
    FinetuningMethod
)


class HPO(Stage):
    def execute(self):
        # todo: some parameters defined in later stages are useful for predicting hp on current stage
        #       probably need to rework HPO, method_selection, finetuning stages
        # TODO: which parameters is HPO and which is method selection?
        # training hyperparams
        self.config.configure('learning_rate', 1e-4)
        self.config.configure('epochs', 20)
        self.config.configure('optimizer', ModelOptimizer.ADAM)
        # configuring scheduler ...

        if self.config.add_linear_part:
            self.config.configure('linear_part_dropout', 0.1)
            self.config.configure('additional_learning_rate', 1e-3)
            self.config.configure('additional_optimizer', ModelOptimizer.ADAM)

        if self.config.finetuning_method == FinetuningMethod.LORA:
            lora_task = TASK_TO_LORA_TASK_MAPPING[self.config.task]

            # todo: process exceptions
            self.config.configure('lora_task', lora_task)
            self.config.configure('lora_r', 8)
            self.config.configure('lora_alpha', 32)
            self.config.configure('lora_dropout', 0.1)

    def validate(self) -> bool:
        pass
