from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.NERModel import RobertaLightningNER
from src.NERDataModule import NERDataModule, initialize_caches


def create_wandb_name(hparams):
    wandb_base_name = 'BitFit' if hparams.bitfit == 'true' else 'Fully'
    dataset = str(hparams.dataset)
    func_type = int(hparams.func_type)
    preprocessing = 'SAME' if func_type == 1 else ('IGNORE' if func_type == 2 else 'INSIDE')
    batch = str(hparams.batch_size)

    return 'NER_' + wandb_base_name \
            + "-DataSet:" + dataset \
            + "-Preprocessing:" + preprocessing \
            + "-Batch:" + batch


def main(hparams):
    # float precision
    torch.set_float32_matmul_precision('high')

    # init logging
    offline_run = True if hparams.offline == 'true' else False
    wandb_logger = WandbLogger(
        log_model=False,
        project="MNLP-MasakhanerNER", name=create_wandb_name(hparams), checkpoint_name=create_wandb_name(hparams),
        save_dir=hparams.save_dir,
        offline=offline_run
    )

    # init the building blocks
    data_config = 'en' if hparams.dataset == 'wikiann' else None
    dataset = 'wikiann' if data_config == 'en' else 'conll2003'
    datamodule = NERDataModule(train_dataset=dataset,
                               config=data_config,
                               func_type=int(hparams.func_type),
                               batch_size=int(hparams.batch_size))
    module = RobertaLightningNER(embedding_dim=768, ner_tags=7,
                                 learning_rate=2e-5, weight_decay=0.05,
                                 name_mapping=datamodule.name_mapping)
    module.model.setup_bitfit_fine_tuning() if hparams.bitfit == 'true' else module.model.setup_fully_fine_tuning()

    # trainer (https://lightning.ai/docs/pytorch/stable/common/trainer.html)
    epochs = 5 if dataset == 'wikiann' else 10
    trainer = pl.Trainer(accelerator=hparams.accelerator,
                         devices=hparams.devices,
                         max_epochs=epochs,
                         logger=wandb_logger,
                         default_root_dir=hparams.save_dir)

    # fit the model (and evaluate on validation data as defined)
    trainer.fit(model=module, datamodule=datamodule)

    # test model
    trainer.test(model=module, datamodule=datamodule)

    # finish run for current logger, enable creation of a new logger in same process
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    initialize_caches()

    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default='auto')
    parser.add_argument("--dataset", default='wikiann') # conll2003 or wikiann
    parser.add_argument("--func_type", default='2') # 0: subtoken inside label, 1: subtoken same label, 2: ignore subtoken
    parser.add_argument("--batch_size", default='16')
    parser.add_argument("--bitfit", default='false') # only bitfit if set to true else fully
    parser.add_argument("--save_dir", default='/datadisk1/av11/downloads/modelcheckpoints')
    parser.add_argument("--offline", default='true') # syncing wandb while running
    args = parser.parse_args()

    main(args)
