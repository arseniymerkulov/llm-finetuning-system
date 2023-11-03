import pytorch_lightning as pl
import torchmetrics
import torch
import glob
import re


from core.main_process.pipeline import Stage
from core.finetuning.finetuning import NLPModel
from core.configuration import TASK_TO_METRICS_MAPPING, Metric, Task


class NLPEvaluatingModel(NLPModel):
    def __init__(self):
        # todo: research problems in comments
        # todo: only cpu support
        super().__init__()
        self.metrics = [self._create_metric(metric['metric']).cuda() for metric in self.config.metrics]

    def _create_metric(self, metric: Metric):
        if metric == Metric.PERPLEXITY:
            # do i need to specify ignore_index for special tokens for cleaner score?
            return torchmetrics.text.Perplexity()
        elif metric == Metric.BLEU:
            return torchmetrics.text.BLEUScore()
        elif metric == Metric.ACCURACY:
            return torchmetrics.classification.Accuracy(task='multiclass', num_classes=self.config.num_classes)
        else:
            raise AssertionError(f'metric class "{metric.value}" does not supported')

    def _preprocess_input_for_metric(self, metric: Metric, batch):
        batch, labels = self._preprocess_input(batch)

        if metric == Metric.BLEU:
            preds = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.config.tokenizer_max_length,
                repetition_penalty=3.0,
                temperature=1.0,
                early_stopping=True)

            preds = self.config.tokenizer.batch_decode(preds)
            labels = self.config.tokenizer.batch_decode(labels)
        else:
            output = self.model.forward(**batch)
            loss, preds = self._preprocess_output(output, labels)

        return preds, labels

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        metrics_dict_for_logging = {}

        for i, metric in enumerate(self.metrics):
            preds, labels = self._preprocess_input_for_metric(self.config.metrics[i]['metric'], batch)

            metric.update(preds, labels)
            metric_value = float(metric.compute().detach().cpu().numpy())
            metrics_dict_for_logging[self.config.metrics[i]['metric'].value] = metric_value
            self.config.metrics[i]['value'] = metric_value

        self.logger.log_metrics(metrics_dict_for_logging)


class Evaluating(Stage):
    def execute(self):
        # todo: cycle between finetuning and evaluation

        self.config.configure('metrics', [{'metric': metric} for metric in TASK_TO_METRICS_MAPPING[self.config.task]])
        self.config.configure('checkpoints_dir', f'{self.config.project_dir}/checkpoints')
        self.config.configure('model', NLPEvaluatingModel())

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
