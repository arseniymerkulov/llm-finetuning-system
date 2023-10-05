from core.main_process.pipeline import Stage
from core.configuration.hyperparams import TASK_TO_LORA_TASK_MAPPING
from core.configuration.configuration import ModelOptimizer, FinetuningMethod


class HPO(Stage):
    def execute(self):
        # TODO: which parameters is HPO and which is method selection?

        # training hyperparams
        self.config.configure('learning_rate', 1e-3)
        self.config.configure('epochs', 10)
        self.config.configure('optimizer', ModelOptimizer.ADAM)
        # configuring scheduler ...

        if self.config.finetuning_method == FinetuningMethod.LORA:
            # configuring LORA parameters ...

            lora_task = TASK_TO_LORA_TASK_MAPPING[self.config.task]

            # todo: process exceptions
            self.config.configure('lora_task', lora_task)
            self.config.configure('lora_r', 8)
            self.config.configure('lora_alpha', 32)
            self.config.configure('lora_dropout', 0.1)

    def validate(self) -> bool:
        pass
