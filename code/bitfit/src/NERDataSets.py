from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset, load_from_disk, concatenate_datasets, disable_caching, DownloadMode

import os
from dotenv import load_dotenv

disable_caching()
load_dotenv(dotenv_path=".env")

# used for everything so ~static
TOKENIZER = AutoTokenizer.from_pretrained("xlm-roberta-base", cache_dir=os.getenv("CACHE_DIR"))
COLLATOR = DataCollatorForTokenClassification(tokenizer=TOKENIZER)  # also pads labels accordingly


"""
see https://huggingface.co/learn/nlp-course/chapter7/2 for alignment implementation
see https://github.com/fdschmidt93/SLICER/blob/224dddfdf75d14a87fe6b084ef485dafb1857048/trident-xtreme/src/tasks/token_classification/processor/__init__.py#L217 for remap motivation

using three different variants 
    -> subtokens have same label (only matters for B- tags)
    -> subtokens are ignored
    -> subtokens are always a respective inside label 
note: -100 is ignored in torch cross entropy loss
note: labels greater than 6 are remapped to 0 as they do not exist in wikiann
      and they do not correspond when comparing conll2003/masakhaner
note: using disk to save and load datasets (less cache wrangling) --> see NERDataModule#initialize_caches
"""
def align_labels_with_tokens_same_label(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else (labels[word_id] if labels[word_id] < 7 else 0)
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id] if labels[word_id] < 7 else 0
            new_labels.append(label)
    return new_labels

def align_labels_with_tokens_ignore_label(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else (labels[word_id] if labels[word_id] < 7 else 0)
            new_labels.append(label)
        else:
            # Special token or sub-token
            new_labels.append(-100)
    return new_labels

def align_labels_with_tokens_inside_label(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else (labels[word_id] if labels[word_id] < 7 else 0)
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            label = label if label < 7 else 0
            new_labels.append(label)
    return new_labels

def tokenize_and_align_labels(examples, align_labels_with_tokens):
    tokenized_inputs = TOKENIZER(
        examples["tokens"], truncation=True, is_split_into_words=True, padding=False, max_length=256
    )

    all_labels = examples['ner_tags']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

def tokenize_and_align_labels_wrapper_same_label(examples):
    return tokenize_and_align_labels(examples, align_labels_with_tokens_same_label)

def tokenize_and_align_labels_wrapper_ignore_label(examples):
    return tokenize_and_align_labels(examples, align_labels_with_tokens_ignore_label)

def tokenize_and_align_labels_wrapper_inside_label(examples):
    return tokenize_and_align_labels(examples, align_labels_with_tokens_inside_label)


class BaseDataSet:
    def __init__(self,
                 name="conll2003",
                 config=None,
                 func_type=0) -> None:
        super().__init__()
        self.dataset_name = name

        match func_type:
            case 1:
                token_align = tokenize_and_align_labels_wrapper_same_label
                self.preprocessing = 'SAME'
            case 2:
                token_align = tokenize_and_align_labels_wrapper_ignore_label
                self.preprocessing = 'IGNORE'
            case _:
                token_align = tokenize_and_align_labels_wrapper_inside_label
                self.preprocessing = 'INSIDE'

        if not len(os.listdir(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/' + self.dataset_name)) == 0:
            self.load()
        else:
            self.dataset = load_dataset(name, config, cache_dir=os.getenv("CACHE_DIR"), download_mode=DownloadMode.FORCE_REDOWNLOAD)
            self.tokenized_datasets = self.dataset.map(token_align, batched=True, remove_columns=self.dataset['train'].column_names, load_from_cache_file=False)
            self.tokenized_datasets["train"].set_format("pt")
            self.tokenized_datasets["validation"].set_format("pt")
            self.tokenized_datasets["test"].set_format("pt")
            self.save()

    def save(self):
        if len(os.listdir(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/' + self.dataset_name)) == 0:
            self.dataset.save_to_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/' + self.dataset_name)
            self.tokenized_datasets.save_to_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + self.dataset_name)

    def load(self):
        self.dataset = load_from_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/' + self.dataset_name)
        self.tokenized_datasets = load_from_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + self.dataset_name)


class BaseMultipleDataSets:
    def __init__(self,
                 base_name='masakhaner',
                 configs=["amh", "hau", "ibo", "kin", "lug", "luo", "pcm", "swa", "wol", "yor"],
                 func_type=0) -> None:
        super().__init__()
        self.multiple_datasets_names = configs

        match func_type:
            case 1:
                token_align = tokenize_and_align_labels_wrapper_same_label
                self.preprocessing = 'SAME'
            case 2:
                token_align = tokenize_and_align_labels_wrapper_ignore_label
                self.preprocessing = 'IGNORE'
            case _:
                token_align = tokenize_and_align_labels_wrapper_inside_label
                self.preprocessing = 'INSIDE'

        self.multiple_datasets = []
        for name in self.multiple_datasets_names:
            if not len(os.listdir(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + name)) == 0:
                self.multiple_datasets.append(self.load(name))
            else:
                dataset = load_dataset(base_name, name, cache_dir=os.getenv("CACHE_DIR"), download_mode=DownloadMode.FORCE_REDOWNLOAD)
                dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
                dataset = dataset.map(token_align, batched=True, remove_columns=dataset.column_names, load_from_cache_file=False)
                dataset.set_format('pt')
                self.multiple_datasets.append(dataset)
                self.save(dataset, name)

    def save(self, dataset, name):
        if len(os.listdir(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + name)) == 0:
            dataset.save_to_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + name)

    def load(self, dataset_name):
        return load_from_disk(os.getenv("CACHE_DIR") + '/' + self.preprocessing + '/tokenized/' + dataset_name)
