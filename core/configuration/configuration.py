from peft import TaskType
from typing import Literal
from enum import Enum
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
        self.dataset_instances: int = 0
        self.dataset_partition: int = 0
        self.dataset_balance: dict = {}
        self.dataset_need_balance: bool = True
        self.dataset_sample_min_length: int = 0
        self.dataset_sample_avg_length: int = 0
        self.dataset_sample_max_length: int = 0
        self.num_classes: int = 0
        self.categories: list[str] = []

        self.model_alias: str = ''
        self.model_auto_class: ModelAutoClass = ModelAutoClass.CAUSAL_LM
        self.model_parameters: int = 0
        self.model_trainable_parameters: int = 0
        self.model_layers: int = 0

        self.pretrain_model = None

        self.tokenizer = None
        self.tokenizer_max_length: int = 0
        self.tokenizer_vocab_size: int = 0

        self.X: list[str] = []
        self.Y: list[str or int] = []

        self.validation_dataset_size: float = 0.0
        self.test_dataset_size: float = 0.0

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

        self.train_dataloader = None
        self.validation_dataloader = None
        self.test_dataloader = None

        self.batch_size: int = 0
        # unused field
        self.num_workers: int = 0

        self.finetuning_method: FinetuningMethod = FinetuningMethod.LORA
        self.add_linear_part: bool = False
        self.linear_part_dropout: float = 0.0
        # parameters for quantization ...

        self.loss_method: LossMethod = LossMethod.INTEGRATED
        self.learning_rate: float = 0.0
        self.epochs: int = 0
        self.optimizer: ModelOptimizer = ModelOptimizer.ADAM
        # self.scheduler: ModelScheduler = ...

        self.lora_task: TaskType = TaskType.CAUSAL_LM
        self.lora_r: float = 0
        self.lora_alpha: float = 0
        self.lora_dropout: float = 0.0

        self.wandb_token_path: str = ''
        self.checkpoints_dir: str = ''
        self.best_checkpoint_path: str = ''
        self.model = None

        self.metrics: list[dict] = []
        self.examples: list[str] = []

        self.configured = self.__dict__.copy()
        self.forced_fields: list[str] = []
        for key in self.configured:
            self.configured[key] = False

        self.status: dict = {}
        self.configure_status('stage', '')
        self.configure_status('phase', '')
        self.configure_status('error', None)
        self.configure_status('approved', False)

    def get_log(self):
        log = {}

        for key in self.configured:
            field = getattr(self, key)

            if isinstance(field, str) or isinstance(field, int) or isinstance(field, float) or isinstance(field, bool):
                log[key] = field

            if isinstance(field, Enum):
                log[key] = field.name

        # log specific lists separately
        log['categories'] = self.categories
        log['dataset_balance'] = self.dataset_balance

        return log

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

    def configure(self, attribute, value, forced=False):
        assert hasattr(self, attribute), f'there is no such field "{attribute}" in the configuration'
        assert isinstance(value, getattr(self, attribute).__class__) or getattr(self, attribute) is None, \
            f'value for field "{attribute}" is not {getattr(self, attribute).__class__} type'

        if attribute in self.forced_fields and self.configured[attribute] and not forced:
            Configuration.logger.info(f'field "{attribute}" is forced and can not be configured')
            return

        if forced and attribute not in self.forced_fields:
            self.forced_fields.append(attribute)

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
