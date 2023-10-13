from peft import TaskType
from typing import Literal
import logging
import time


from core.settings import Settings
from .hyperparams import (
    PipelineSetup,
    Task,
    DatasetStorageFormat,
    FinetuningMethod,
    ModelOptimizer,
    LossMethod,
    ModelAutoClass
)


class Configuration:
    # todo: setting level on app level
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    settings = Settings.get_instance()
    _instance = None

    def __init__(self):
        self.project: str = ''
        self.project_dir: str = ''

        self.pipeline_setup: PipelineSetup = PipelineSetup.FULL
        self.task: Task = Task.CAUSAL_LM

        # self.dataset_collecting_type: enum
        # self.transformers_token_path: str
        self.kaggle_token_path: str = ''
        self.dataset_alias: str = ''
        self.dataset_dir: str = ''

        self.dataset_file: str = ''
        self.dataset_storage_format: DatasetStorageFormat = DatasetStorageFormat.TABLE
        self.dataset_table_columns: list[str] = []
        self.dataset_partition: int = 0
        self.dataset_balance: bool = True
        self.num_classes: int = 2
        self.categories: list[str] = []

        self.model_alias: str = ''
        self.model_auto_class: ModelAutoClass = ModelAutoClass.CAUSAL_LM
        self.pretrain_model = None

        self.tokenizer = None
        self.tokenizer_max_length: int = 128

        self.X: list[str] = []
        self.Y: list[str or int] = []

        self.validation_dataset_size: float = 0.3
        self.test_dataset_size: float = 0.1

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None

        self.batch_size: int = 12
        self.num_workers: int = 2

        self.finetuning_method: FinetuningMethod = FinetuningMethod.LORA
        self.add_linear_part: bool = False
        self.linear_part_dropout: float = 0.1
        self.linear_part_power: int = 2
        # parameters for linear part ...
        # parameters for quantization ...

        self.loss_method: LossMethod = LossMethod.INTEGRATED
        self.learning_rate: float = 1e-3
        self.epochs: int = 10
        self.optimizer: ModelOptimizer = ModelOptimizer.ADAM
        # self.scheduler: ModelScheduler = ...

        self.lora_task: TaskType = TaskType.CAUSAL_LM
        self.lora_r: float = 8
        self.lora_alpha: float = 32
        self.lora_dropout: float = 0.1

        self.wandb_token_path: str = ''
        self.checkpoints_dir: str = ''
        self.best_checkpoint_path: str = ''
        self.model = None

        self.metrics: list[dict] = []

        self.configured = self.__dict__.copy()
        for key in self.configured:
            self.configured[key] = False

        self.status: dict = {}
        self.configure_status('stage', '')
        self.configure_status('phase', '')
        self.configure_status('error', None)
        self.configure_status('approved', False)

    def configure_status(self, attribute: Literal['stage', 'phase', 'error', 'approved'], value):
        status = {
            'stage': self.status['stage'] if 'stage' in self.status else '',
            'phase': self.status['phase'] if 'phase' in self.status else '',
            'success': self.status['success'] if 'success' in self.status else True,
            'error': self.status['error'] if 'error' in self.status else None,
            'approved': False,
            attribute: value
        }

        if attribute == 'error' and value is not None:
            status['success'] = False

        self.status = status

    def configure(self, attribute, value):
        assert hasattr(self, attribute), f'there is no such field "{attribute}" in the configuration'
        self.__setattr__(attribute, value)
        self.configured[attribute] = True
        Configuration.logger.info(f'field "{attribute}" configured')

    def wait_for_approval(self, timeout=settings.stage_approval_timeout):
        Configuration.logger.info(f'waiting for stage "{self.status["stage"]}" to be approved')
        start_time = time.time()

        while not self.status['approved']:
            assert time.time() - start_time < timeout, \
                f'stage {self.status["stage"]} has not been approved in time ({timeout}s)'
            time.sleep(0.1)

        Configuration.logger.info(f'stage "{self.status["stage"]}" is approved')

    def wait(self, attribute, timeout=settings.field_configure_timeout):
        Configuration.logger.info(f'waiting for field "{attribute}" to be configured')
        start_time = time.time()

        while not self.configured[attribute]:
            assert time.time() - start_time < timeout, f'field {attribute} has not been configured in time ({timeout}s)'
            time.sleep(0.1)
        Configuration.logger.info(f'field "{attribute}" is configured in {time.time() - start_time}s')

    @staticmethod
    def reset():
        Configuration._instance.__init__()

    @staticmethod
    def get_instance():
        if not Configuration._instance:
            Configuration._instance = Configuration()

        return Configuration._instance
