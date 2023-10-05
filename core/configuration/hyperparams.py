from peft import TaskType
from enum import Enum
import torchmetrics


class Task(Enum):
    CLASSIFICATION = 1
    CAUSAL_LM = 2
    SEQ_2_SEQ_LM = 3


class DatasetStorageFormat(Enum):
    CSV_TABLE = 1


class FinetuningMethod(Enum):
    FULL_FINETUNING = 1
    LORA = 2


class ModelOptimizer(Enum):
    SGD = 'SGD'
    ADAM = 'Adam'
    ADAM_W = 'AdamW'


class ModelAutoClass(Enum):
    DEFAULT = 'AutoModel'
    CAUSAL_LM = 'AutoModelForCausalLM'
    SEQ_2_SEQ_LM = 'AutoModelForSeq2SeqLM'


class Metric(Enum):
    ACCURACY = 'Accuracy'
    PERPLEXITY = 'Perplexity'
    ROUGE = 'Rouge'


TASK_TO_AUTO_CLASS_MAPPING = {
    Task.CLASSIFICATION: ModelAutoClass.DEFAULT,
    Task.CAUSAL_LM: ModelAutoClass.CAUSAL_LM,
    Task.SEQ_2_SEQ_LM: ModelAutoClass.SEQ_2_SEQ_LM
}


TASK_TO_LORA_TASK_MAPPING = {
    Task.CLASSIFICATION: TaskType.SEQ_CLS,
    Task.CAUSAL_LM: TaskType.CAUSAL_LM,
    Task.SEQ_2_SEQ_LM: TaskType.SEQ_2_SEQ_LM
}


TASK_TO_METRICS_MAPPING = {
    Task.CLASSIFICATION: [Metric.ACCURACY],
    Task.CAUSAL_LM: [Metric.PERPLEXITY],
    Task.SEQ_2_SEQ_LM: []
}
