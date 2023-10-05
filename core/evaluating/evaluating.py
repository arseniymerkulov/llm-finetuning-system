import pytorch_lightning as pl
import torchmetrics
import torch
import glob
import os


from core.main_process.pipeline import Stage
from core.finetuning.finetuning import NLPModel
from core.configuration.configuration import Configuration
from core.configuration.hyperparams import TASK_TO_METRICS_MAPPING, Metric


class NLPEvaluatingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = Configuration.get_instance()
        self.model = self.config.model
        self.metrics = [self._create_metric(metric['metric']).cuda() for metric in self.config.metrics]

    def _create_metric(self, metric: Metric):
        if metric == Metric.PERPLEXITY:
            # do i need to specify ignore_index for special tokens for cleaner score?
            return torchmetrics.text.Perplexity()
        elif metric == Metric.ACCURACY:
            return torchmetrics.classification.Accuracy(task='multiclass', num_classes=self.config.num_classes)
        else:
            raise AssertionError(f'metric class "{metric.value}" does not supported')

    def _compute_metrics(self):
        metrics_dict_for_logging = {}

        for i, metric in enumerate(self.metrics):
            # if metric == ROUGE -> metric.compute()['rouge1']
            metric_value = float(metric.compute().detach().cpu().numpy())

            metrics_dict_for_logging[self.config.metrics[i]['metric'].value] = metric_value
            self.config.metrics[i]['value'] = metric_value

        self.logger.log_metrics(metrics_dict_for_logging)

    @torch.no_grad()
    def test_step(self, batch, batch_idx: int):
        input_ids, attn_mask, labels = (
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels'],
        )

        # todo: input parameters dependency from model
        # todo: output parameters dependency from model

        output = self.model.forward(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        [metric.update(output.logits, labels) for metric in self.metrics]
        self._compute_metrics()


class Evaluating(Stage):
    def execute(self):
        # todo: assert that checkpoint exists

        self.config.configure('model', NLPModel())
        self.config.configure('checkpoints_dir', f'{self.config.project_dir}/checkpoints')
        self.config.configure('metrics', [{'metric': metric} for metric in TASK_TO_METRICS_MAPPING[self.config.task]])

        if not self.config.configured['best_checkpoint_path']:
            self.config.configure('best_checkpoint_path', self._get_best_model())

        self.config.model.load_state_dict(torch.load(self.config.best_checkpoint_path)['state_dict'])
        self.config.model.eval()

        logger = pl.loggers.WandbLogger(name=self.config.model_alias, project=f'{self.config.project}-evaluating')

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='auto',
            logger=logger
        )

        trainer.test(
            NLPEvaluatingModel(),
            self.config.test_dataloader
        )

    def _get_best_model(self):
        models = glob.glob(f'{self.config.checkpoints_dir}/*/*.ckpt')
        assert len(models), 'empty checkpoints directory'

        # checkpoint filename format specified in the core/settings.py
        losses = [float(os.path.split(model_path)[-1].split('val_loss=')[1].split('.ckpt')[0]) for model_path in models]
        models = [model_path for _, model_path in sorted(zip(losses, models), reverse=True, key=lambda pair: pair[0])]
        return models[0]

    def validate(self) -> bool:
        pass
