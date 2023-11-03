from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import get_peft_model, LoraConfig
import pytorch_lightning as pl
import torch
import os


from core.configuration import Configuration, FinetuningMethod, LossMethod, Task
from core.main_process.pipeline import Stage


class NLPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = Configuration.get_instance()
        self.train_loss, self.val_loss = [], []

        if self.config.add_linear_part:
            self.model = self._create_combine_model(self.config.pretrain_model)
        else:
            self.model = self.config.pretrain_model

        if self.config.loss_method == LossMethod.CROSS_ENTROPY:
            self.loss_method = torch.nn.CrossEntropyLoss()

        if self.config.finetuning_method == FinetuningMethod.FULL_FINETUNING:
            for param in self.model.parameters():
                param.requires_grad = True

        elif self.config.finetuning_method == FinetuningMethod.LORA:
            # todo: gpt-2 and gpt2 aliases exists
            assert any(key in self.config.model_alias for key
                       in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING.keys()), \
                f'LoRa does not have support for model "{self.config.model_alias}", you need to specify layers manually'

            lora = LoraConfig(
                task_type=self.config.lora_task,
                inference_mode=False,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
            )
            self.model = get_peft_model(self.model, lora)
            self.model.print_trainable_parameters()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def _create_combine_model(self, pretrain_model):
        class CombinedModel(torch.nn.Module):
            def __init__(self, pretrain_model, linear_part):
                super().__init__()
                self.pretrain_model = pretrain_model
                self.linear_part = linear_part

            def forward(self, *args, **kwargs):
                output = self.pretrain_model(*args, **kwargs)
                return self.linear_part(output.last_hidden_state[:, 0, :])

        # todo: assert that output dims can be retrieved
        *_, last_layer = self.config.pretrain_model.children()
        pretrain_output_dims = last_layer.dense.out_features

        # find powers of 2 with condition: output_dims >= lower_estimate >= top_estimate >= num_classes
        lower_estimate = 1 << (pretrain_output_dims.bit_length() - 1)
        top_estimate = 1 << (self.config.num_classes - 1).bit_length()
        second_linear_layer_out = int(lower_estimate if top_estimate == lower_estimate else lower_estimate / 2)

        linear_part = [
            torch.nn.Dropout(self.config.linear_part_dropout),
            torch.nn.Linear(pretrain_output_dims, lower_estimate),
            torch.nn.BatchNorm1d(lower_estimate),
            torch.nn.ReLU(),
            torch.nn.Linear(lower_estimate, second_linear_layer_out),
            torch.nn.ReLU(),
            torch.nn.Linear(second_linear_layer_out, self.config.num_classes),
            torch.nn.LogSoftmax(dim=1)
        ]

        return CombinedModel(pretrain_model, torch.nn.Sequential(*linear_part))

    def _preprocess_input(self, batch):
        labels = batch['labels']

        if self.config.loss_method != LossMethod.INTEGRATED:
            # todo: research for possible problems with deleting data from batch. Do dataloader clone batches?
            del batch['labels']

        # todo: useless for now, mb delete from dataset
        if self.config.task != Task.CLASSIFICATION:
            del batch['decoder_attention_mask']
            del batch['decoder_input_ids']

        return batch, labels

    def _preprocess_output(self, output, labels):
        if self.config.loss_method == LossMethod.INTEGRATED:
            return output.loss, output.logits

        if self.config.add_linear_part:
            return self.loss_method(output, labels), output

        raise AssertionError(f'unsupported combination of parameters: loss_method={self.config.loss_method.name} '
                             f'and add_linear_part={self.config.add_linear_part}')

    def training_step(self, batch, batch_idx):
        batch, labels = self._preprocess_input(batch)
        output = self.model.forward(**batch)
        loss, _ = self._preprocess_output(output, labels)

        self.log('train_loss', loss)
        self.train_loss.append(loss.detach().cpu())

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        batch, labels = self._preprocess_input(batch)
        output = self.model.forward(**batch)
        loss, _ = self._preprocess_output(output, labels)

        self.log('val_loss', loss)
        self.val_loss.append(loss.detach().cpu())

    def configure_optimizers(self):
        assert any([parameter.requires_grad for parameter in self.model.parameters()]), \
            'all weights in model are frozen'
        assert hasattr(torch.optim, self.config.optimizer.value), 'invalid optimizer name'
        # todo: different lr for different model parts
        optimizer = getattr(torch.optim, self.config.optimizer.value)(self.model.parameters(),
                                                                      lr=self.config.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        outputs = self.train_loss

        loss = torch.stack([x for x in outputs]).mean()
        print(f'train loss = {loss:.8f}\n')
        self.train_loss = []

    def on_validation_epoch_end(self):
        outputs = self.val_loss

        loss = torch.stack([x for x in outputs]).mean()
        print(f'val loss = {loss:.8f}\n')
        self.val_loss = []


class Finetuning(Stage):
    def execute(self):
        self.config.configure('checkpoints_dir', f'{self.config.project_dir}/checkpoints')

        if not os.path.exists(self.config.checkpoints_dir):
            os.mkdir(self.config.checkpoints_dir)

        logger = pl.loggers.WandbLogger(name=self.config.model_alias, project=self.config.project)
        self.config.configure('model', NLPModel())

        # todo: estimate epochs and patience in HPO
        early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            dirpath=self.config.checkpoints_dir,
            filename=f'{self.config.model_alias}{self.settings.checkpoint_postfix}',
            save_top_k=1
        )

        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='auto',
            logger=logger,
            callbacks=[
                early_stopping_callback,
                model_checkpoint_callback
            ]
        )

        trainer.fit(
            self.config.model,
            self.config.train_dataloader,
            val_dataloaders=[self.config.validation_dataloader]
        )

        self.config.configure('best_checkpoint_path', model_checkpoint_callback.best_model_path)

    def validate(self) -> bool:
        pass
