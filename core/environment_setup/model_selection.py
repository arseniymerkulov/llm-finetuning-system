import transformers


from core.main_process.pipeline import Stage
from core.configuration.hyperparams import TASK_TO_AUTO_CLASS_MAPPING


class ModelSelection(Stage):
    def execute(self):
        # todo: assert model alias exists (on previous stage?)
        self.config.wait('model_alias')

        # todo: exception processing
        auto_class = TASK_TO_AUTO_CLASS_MAPPING[self.config.task].value
        model = getattr(transformers, auto_class).from_pretrained(self.config.model_alias)

        self.config.configure('pretrain_model', model)

    def validate(self) -> bool:
        pass