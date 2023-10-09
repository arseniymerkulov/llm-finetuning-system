import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


from core.main_process.pipeline import Stage
from core.configuration import Configuration


class NLPDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()

        self.config = Configuration.get_instance()
        self.tokenizer = self.config.tokenizer

        self.tokenized_Y = None
        self.tokenized_X = self.tokenizer(X,
                                          return_tensors='pt',
                                          padding='max_length',
                                          truncation=True,
                                          max_length=self.config.tokenizer_max_length)

        if isinstance(Y[0], str):
            self.tokenized_Y = self.tokenizer(Y,
                                              return_tensors='pt',
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.config.tokenizer_max_length)

            self.targets = self.tokenized_Y['input_ids'].clone()
        else:
            self.targets = torch.Tensor(Y)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, item):
        item = {
            'input_ids': self.tokenized_X['input_ids'][item],
            'labels': self.targets[item]
        }

        if self.tokenized_Y:
            item['decoder_input_ids'] = self.tokenized_Y['input_ids'][item]
            item['decoder_attention_mask'] = self.tokenized_Y['input_ids'][item]

        return item


class DataTokenizing(Stage):
    def execute(self):
        self.config.configure('tokenizer', AutoTokenizer.from_pretrained(self.config.model_alias))
        self.config.configure('tokenizer_max_length', 512)

        if self.config.tokenizer.pad_token is None:
            self.config.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.config.pretrain_model.resize_token_embeddings(len(self.config.tokenizer))

        self.config.configure('validation_dataset_size', 0.3)
        self.config.configure('test_dataset_size', 0.1)
        self.config.configure('batch_size', 4)

        x_train, x_test, y_train, y_test = train_test_split(self.config.X,
                                                            self.config.Y,
                                                            test_size=self.config.test_dataset_size)

        x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                          y_train,
                                                          test_size=self.config.validation_dataset_size)

        self.config.configure('train_dataset', NLPDataset(x_train, y_train))

        self.config.configure('validation_dataset', NLPDataset(x_val, y_val))

        self.config.configure('test_dataset', NLPDataset(x_test, y_test))

        self.config.configure('train_dataloader', DataLoader(self.config.train_dataset,
                                                             batch_size=self.config.batch_size,
                                                             shuffle=True))

        self.config.configure('validation_dataloader', DataLoader(self.config.validation_dataset,
                                                                  batch_size=self.config.batch_size,
                                                                  shuffle=False))

        self.config.configure('test_dataloader', DataLoader(self.config.test_dataset,
                                                            batch_size=self.config.batch_size,
                                                            shuffle=False))

    def validate(self) -> bool:
        pass
