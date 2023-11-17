from core.main_process.pipeline import Stage
from core.configuration import FinetuningMethod, LossMethod, Task


class FinetuningMethodSelection(Stage):
    def execute(self):
        self.config.configure('finetuning_method', FinetuningMethod.LORA)
        # configuring linear part, quantization

        if self.config.task == Task.CLASSIFICATION:
            # todo: only if model forward cant handle 'labels', for example rubert
            self.config.configure('add_linear_part', True)
            self.config.configure('loss_method', LossMethod.CROSS_ENTROPY)
        else:
            self.config.configure('loss_method', LossMethod.INTEGRATED)

    def validate(self) -> bool:
        pass
