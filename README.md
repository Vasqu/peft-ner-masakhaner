# Multilingual NLP


## Introduction

This project implements fine-tuning of a multilingual transformer ("*xlm-roberta-base*") for the named entity recognition task (NER) in various ways. First, a simple complete fine-tuning and then a PEFT variant, more specifically BitFit. The goal is to compare the effectiveness of the implemented tweak. 


## Roadmap
- Model implementation:
    - "*xlm-roberta-base*" as encoder/body (via ```AutoModel.from_pretrained```)
    - Preprocessing needed
    - Own classification head using cross entropy loss
    - Reference: [Token classification guide by huggingface](https://huggingface.co/learn/nlp-course/chapter7/2?fw=pt)
- Train with AdamW 
    - Fully fine-tuning or [BitFit](https://aclanthology.org/2022.acl-short.1.pdf)
    - Learning rate ~ 2e-5 
    - Weight decay ~ 0.05
    - Sets:
        - [ConLL](https://huggingface.co/datasets/conll2003) ~ 10 epochs
        - [WikiAnn](https://huggingface.co/datasets/wikiann) ~ 5 epochs
- Eval
    - Micro F1 on last checkpoint
    - Set(s):
        - [Languages part of MasakhaNER](https://huggingface.co/datasets/masakhaner)
- Reference for using multiple sets: [Torch Lightning - Managing Data](https://lightning.ai/docs/pytorch/LTS/guides/data.html)


# Usage
Just run the [NERRun.py](code/bitfit/src/NERRun.py) script. Might need a login to your wandb account before (e.g. via the terminal) and possibly might need a predefined cache structure at a given location (see [NERDataModule.py](code/bitfit/src/NERDataModule.py)). 


## Issues
The project evaluates on a micro f1 basis which heavily favors the outside tag (which is not really desirable). If anyone uses the repo, they should consider using macro f1 instead (or exclude the O tag from evaluation). In my case, it is/was not really necessary as the direct comparison is sufficient to get a first impression on the effectiveness. 

Also, the plot creation is also done in a messy copy paste fashion, don't mind it :P
