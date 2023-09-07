from torch.utils.data import DataLoader
from pytorch_lightning.utilities import CombinedLoader
from src.NERDataSets import BaseDataSet, BaseMultipleDataSets, COLLATOR
from pytorch_lightning import LightningDataModule


class NERDataModule(LightningDataModule):
    def __init__(self,
                 train_dataset="conll2003",
                 config=None,
                 func_type=0,
                 batch_size=32):
        super().__init__()
        self.dataset = BaseDataSet(train_dataset,
                                   config=config,
                                   func_type=func_type)
        masakhaner_configs = ["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"]
        self.masakhaner = BaseMultipleDataSets(base_name='masakhaner',
                                               configs=masakhaner_configs,
                                               func_type=func_type)
        # mapping index to name to get some interpretable results later on
        self.name_mapping = {
            i: masakhaner_configs[i] for i in range(len(masakhaner_configs))
        }
        self.name_mapping[len(masakhaner_configs)] = self.dataset.dataset_name
        self.batch_size = batch_size

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(self.dataset.tokenized_datasets["train"],
                          batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=COLLATOR)

    def val_dataloader(self):
        return DataLoader(self.dataset.tokenized_datasets["validation"],
                          batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=COLLATOR)

    def test_dataloader(self):
        dataloaders = {
            self.masakhaner.multiple_datasets_names[i]: DataLoader(self.masakhaner.multiple_datasets[i],
                                                                   batch_size=self.batch_size,
                                                                   shuffle=False,
                                                                   collate_fn=COLLATOR)
            for i in range(len(self.masakhaner.multiple_datasets))
        }
        dataloaders[self.dataset.dataset_name] = DataLoader(self.dataset.tokenized_datasets["test"],
                                                            batch_size=self.batch_size,
                                                            shuffle=False,
                                                            collate_fn=COLLATOR)

        # cannot use parallel batches for now
        # see https://github.com/Lightning-AI/lightning/issues/16830 for reference
        return CombinedLoader(dataloaders, mode='sequential')


"""
introduced due to the following issue:
--> https://github.com/huggingface/datasets/issues/5736
    
expects the following cache dir structure:
    cache_dir
    |
    ...
    |
    -- IGNORE
        |
        -- conll2003
        |
        -- tokenized
            |
            -- amh
            |
            -- conll2003
            |
            ...
            |
            -- wikiann
            |
            ...
            |
            ...
        |
        -- wikiann
    |
    -- INSIDE
        |
        same as IGNORE structure
    |
    -- SAME
        |
        same as IGNORE structure
        
run it once before starting the run(s)
"""
def initialize_caches():
    func_types = [0, 1, 2]
    datasets = ['conll2003', 'wikiann']

    for func in func_types:
        for dataset in datasets:
            data_config = 'en' if dataset == 'wikiann' else None
            dataset = 'wikiann' if data_config == 'en' else 'conll2003'
            datamodule = NERDataModule(train_dataset=dataset,
                                       config=data_config,
                                       func_type=func,
                                       batch_size=16)
