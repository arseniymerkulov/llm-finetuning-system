from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft import get_peft_model, LoraConfig
import pytorch_lightning as pl
import torch
import os


from core.configuration.configuration import Configuration, FinetuningMethod
from core.main_process.pipeline import Stage


class NLPModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.config = Configuration.get_instance()
        self.model = self.config.pretrain_model
        self.train_loss, self.val_loss = [], []

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

    def training_step(self, batch, batch_idx):
        input_ids, attn_mask, labels = (
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels'],
        )

        # todo: input parameters dependency from model
        # todo: output parameters dependency from model

        output = self.model.forward(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = output.loss

        self.log('train_loss', loss)
        self.train_loss.append(loss.detach().cpu())

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx: int):
        input_ids, attn_mask, labels = (
            batch['input_ids'],
            batch['attention_mask'],
            batch['labels'],
        )

        # todo: input parameters dependency from model
        # todo: output parameters dependency from model

        output = self.model.forward(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = output.loss
        self.log('val_loss', loss)
        self.val_loss.append(loss.detach().cpu())

    def configure_optimizers(self):
        assert any([parameter.requires_grad for parameter in self.model.parameters()]), \
            'all weights in model are frozen'
        assert hasattr(torch.optim, self.config.optimizer.value), 'invalid optimizer name'
        optimizer = getattr(torch.optim, self.config.optimizer.value)(self.model.parameters(),
                                                                      lr=self.config.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        outputs = self.train_loss

        loss = torch.stack([x for x in outputs]).mean()
        print(f'train loss = {loss:.15f}\n')
        self.train_loss = []

    def on_validation_epoch_end(self):
        outputs = self.val_loss

        loss = torch.stack([x for x in outputs]).mean()
        print(f'val loss = {loss:.15f}\n')
        self.val_loss = []


class Finetuning(Stage):
    def execute(self):
        self.config.configure('checkpoints_dir', f'{self.config.project_dir}/checkpoints')

        if not os.path.exists(self.config.checkpoints_dir):
            os.mkdir(self.config.checkpoints_dir)

        logger = pl.loggers.WandbLogger(name=self.config.model_alias, project=self.config.project)
        self.config.configure('model', NLPModel())

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
