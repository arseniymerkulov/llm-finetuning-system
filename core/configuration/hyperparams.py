from peft import TaskType
from enum import Enum


class PipelineSetup(Enum):
    FULL = 1
    EVALUATION = 2
    DATA_ANALYSIS = 3


class Task(Enum):
    CLASSIFICATION = 1
    CAUSAL_LM = 2
    SEQ_2_SEQ_LM = 3


class DatasetStorageFormat(Enum):
    TABLE = 1


class FinetuningMethod(Enum):
    FULL_FINETUNING = 1
    LORA = 2


class LossMethod(Enum):
    INTEGRATED = 'Integrated'
    CROSS_ENTROPY = 'CrossEntropyLoss'


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
    RECALL = 'Recall'
    PRECISION = 'Precision'
    F1_SCORE = 'F1Score'
    PERPLEXITY = 'Perplexity'
    ROUGE = 'ROUGEScore'
    BLEU = 'BLEUScore'


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
    Task.CLASSIFICATION: [Metric.ACCURACY, Metric.RECALL, Metric.PRECISION, Metric.F1_SCORE],
    Task.CAUSAL_LM: [Metric.PERPLEXITY],
    Task.SEQ_2_SEQ_LM: [Metric.BLEU, Metric.ROUGE]
}
