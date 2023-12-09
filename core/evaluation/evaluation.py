from transformers import GenerationConfig
import pytorch_lightning as pl
import torchmetrics
import torch
import glob
import re


from core.main_process.pipeline import Stage
from core.finetuning.finetuning import NLPModel
from core.configuration import TASK_TO_METRICS_MAPPING, Metric, Task


class NLPEvaluationModel(NLPModel):
    def __init__(self):
        # todo: add cpu support
        super().__init__()
        self.metrics = [self._create_metric(metric['metric']).cuda() for metric in self.config.metrics]

    def _create_metric(self, metric: Metric):
        if metric == Metric.ACCURACY:
            return torchmetrics.classification.Accuracy(task='multiclass',
                                                        num_classes=self.config.num_classes,
                                                        average='macro')
        elif metric == Metric.RECALL:
            return torchmetrics.classification.Recall(task='multiclass',
                                                      num_classes=self.config.num_classes,
                                                      average='macro')
        elif metric == Metric.PRECISION:
            return torchmetrics.classification.Precision(task='multiclass',
                                                         num_classes=self.config.num_classes,
                                                         average='macro')
        elif metric == Metric.F1_SCORE:
            return torchmetrics.classification.F1Score(task='multiclass',
                                                       num_classes=self.config.num_classes,
                                                       average='macro')
        elif metric == Metric.PERPLEXITY:
            # do i need to specify ignore_index for special tokens for cleaner score?
            return torchmetrics.text.Perplexity()
        elif metric == Metric.ROUGE:
            return torchmetrics.text.ROUGEScore()
        elif metric == Metric.BLEU:
            return torchmetrics.text.BLEUScore()
        else:
            raise AssertionError(f'metric class "{metric.value}" does not supported')

    def _generate(self, input_ids, attention_mask):
        # todo: estimate generation parameters

        try:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GenerationConfig.from_pretrained(self.config.model_alias)
            )
        except EnvironmentError:
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.config.tokenizer_max_length,
                repetition_penalty=3.0,
                temperature=1.0,
                early_stopping=True)

    def _compute_metrics(self, batch, labels):
        # todo: why last logged value is not equal to last computed metric in console?
        output = self.model.forward(**batch)
        _, preds = self._preprocess_output(output, labels)
        preds_decoded, labels_decoded = None, None

        metrics_dict_for_logging = {}
        metrics = [metric['metric'] for metric in self.config.metrics]

        if Metric.ROUGE in metrics or Metric.BLEU in metrics:
            preds_decoded = self._generate(batch['input_ids'], batch['attention_mask'])
            preds_decoded = self.config.tokenizer.batch_decode(preds_decoded)
            labels_decoded = self.config.tokenizer.batch_decode(labels)

        for i, metric in enumerate(self.metrics):
            metric_type = self.config.metrics[i]['metric']

            # different logic for metrics with text arguments
            if metric_type == Metric.ROUGE or metric_type == Metric.BLEU:
                metric.update(preds_decoded, labels_decoded)
            else:
                metric.update(preds, labels)

            # different logic for metrics with dict output
            if metric_type == Metric.ROUGE:
                metric_dict = metric.compute()
                metric_dict = {key: value.detach().cpu().numpy() for key, value in
                               zip(metric_dict.keys(), metric_dict.values())}
            else:
                metric_dict = {metric_type.value: float(metric.compute().detach().cpu().numpy())}

            self.config.metrics[i]['value'] = metric_dict
            for key in metric_dict:
                metrics_dict_for_logging[key] = metric_dict[key]

        self.logger.log_metrics(metrics_dict_for_logging)

    def _generate_example(self, batch, labels):
        # todo: metrics for classification have same value, looks sus
        input_decoded = self.config.tokenizer.batch_decode(batch['input_ids'])

        if self.config.task != Task.CLASSIFICATION:
            preds = self._generate(batch['input_ids'], batch['attention_mask'])
            preds = self.config.tokenizer.batch_decode(preds)
            labels = self.config.tokenizer.batch_decode(labels)

            return [{
                'input': input_ids.replace(self.config.tokenizer.pad_token, ''),
                'preds': prediction.replace(self.config.tokenizer.pad_token, ''),
                'label': label.replace(self.config.tokenizer.pad_token, '')
            } for input_ids, prediction, label in zip(input_decoded, preds, labels)]
        else:
            output = self.model.forward(**batch)
            _, preds = self._preprocess_output(output, labels)
            preds = [torch.argmax(prediction).detach().cpu().item() for prediction in preds]

            return [{
                'input': input_ids.replace(self.config.tokenizer.pad_token, ''),
                'preds': self.config.categories[prediction],
                'label': self.config.categories[label]
            } for input_ids, prediction, label in zip(input_decoded, preds, labels)]

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        batch, labels = self._preprocess_input(batch)
        self._compute_metrics(batch, labels)

        # todo: think about optimal strategy to 'when' generate examples
        if not len(self.config.examples):
            self.config.configure('examples', self._generate_example(batch, labels))
            self.logger.log_hyperparams({'examples': self.config.examples})


class Evaluation(Stage):
    def execute(self):
        # todo: cycle between finetuning and evaluation

        self.config.configure('metrics', [{'metric': metric} for metric in TASK_TO_METRICS_MAPPING[self.config.task]])
        self.config.configure('checkpoints_dir', f'{self.config.project_dir}/checkpoints')
        self.config.configure('model', NLPEvaluationModel())

        best_model = self.config.best_checkpoint_path if self.config.configured['best_checkpoint_path'] \
            else self._get_best_model()

        if best_model:
            self.config.configure('best_checkpoint_path', best_model)
            self.config.model.load_state_dict(torch.load(self.config.best_checkpoint_path)['state_dict'])

        self.config.model.eval()

        logger = pl.loggers.WandbLogger(name=self.config.model_alias, project=self.config.project)

        trainer = pl.Trainer(
            accelerator='auto',
            logger=logger
        )

        trainer.test(
            self.config.model,
            self.config.test_dataloader
        )

        self.logger.info(f'computed metrics: {self.config.metrics}')

    def _get_best_model(self):
        models = glob.glob(f'{self.config.checkpoints_dir}/*/*.ckpt')
        if not len(models):
            return None

        # checkpoint filename format specified in the core/settings.py
        losses = [float(re.findall(r'[0-9]*[.,]?[0-9]+', model_path)[0]) for model_path in models]
        models = [model_path for _, model_path in sorted(zip(losses, models), reverse=True, key=lambda pair: pair[0])]
        return models[0]

    def validate(self) -> bool:
        pass
