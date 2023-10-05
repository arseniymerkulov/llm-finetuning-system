from core.main_process.pipeline import Stage
from core.configuration.configuration import FinetuningMethod


class FinetuningMethodSelection(Stage):
    def execute(self):
        self.config.configure('finetuning_method', FinetuningMethod.FULL_FINETUNING)
        # configuring linear part, quantization

    def validate(self) -> bool:
        pass
