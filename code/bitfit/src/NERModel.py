import torch
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from transformers import AutoModel

import os
from dotenv import load_dotenv

load_dotenv('.env')


class RobertaNER(torch.nn.Module):
    def __init__(self,
                 embedding_dim, ner_tags,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transformer = AutoModel.from_pretrained("xlm-roberta-base",
                                                     add_pooling_layer=False,
                                                     cache_dir=os.getenv("CACHE_DIR"))
        self.out = torch.nn.Linear(embedding_dim, ner_tags)

    def forward(self, x):
        x = self.transformer(**x)
        x = self.out(x.last_hidden_state)
        return x

    def setup_fully_fine_tuning(self):
        for _, param in self.named_parameters():
            param.requires_grad = True

    def setup_bitfit_fine_tuning(self):
        for name, param in self.named_parameters():
            if name.endswith('bias'):
                param.requires_grad = True
            else:
                param.requires_grad = False


class RobertaLightningNER(LightningModule):
    def __init__(self,
                 embedding_dim=768, ner_tags=9,
                 learning_rate=2e-5, weight_decay=0.05,
                 name_mapping=None):
        # save HP to checkpoints
        self.save_hyperparameters(ignore=['name_mapping'])
        self.name_mapping = name_mapping
        super().__init__()
        # init model
        self.model = RobertaNER(embedding_dim, ner_tags)
        # micro f1
        self.test_f1 = torch.nn.ModuleDict()

    def setup(self, stage):
        # change your model dynamically
        pass

    def forward(self, x):
        return self.model(x)

    # see https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # --> expects: (batch, logits for classes, dimensionality) where dimensionality is our sequence of tokens
    def training_step(self, batch, batch_idx):
        x, y = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, batch['labels']
        logits = self(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, batch['labels']
        logits = self(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)
        self.log_dict({"val_loss": loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx, dataloader_idx):
        # sequentially going through the dataloaders and map statistics to respective dataset name if known
        data_set_name = str(self.name_mapping[dataloader_idx]) if self.name_mapping is not None else 'unknown'
        res = dict()

        # loss
        x, y = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}, batch['labels']
        logits = self(x)
        loss = F.cross_entropy(logits.permute(0, 2, 1), y)
        res['test_loss_' + data_set_name] = loss

        # micro f1
        if data_set_name not in self.test_f1:
            # don't forget to move to the right device
            self.test_f1[data_set_name] = torchmetrics.F1Score(task='multiclass', num_classes=self.hparams.ner_tags, ignore_index=-100, average='micro', multidim_average='global').to(self.device)
        self.test_f1[data_set_name].update(logits.permute(0, 2, 1), y)
        res['test_micro_f1_' + data_set_name] = self.test_f1[data_set_name].compute()

        self.log_dict(res, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        # optimizer is also able to update parameters, hence only pass relevant parameters
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                 lr=self.hparams.learning_rate,
                                 weight_decay=self.hparams.weight_decay)
