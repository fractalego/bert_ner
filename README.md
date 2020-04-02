# Ononotes 5 NER with BERT

This is a simple test of NER with BERT. 
The repo is a placeholder for code to reuse. 
The code is basically BertTagger from [Devlin et al](https://arxiv.org/abs/1810.04805).

## Installation
Please install [git-lfs](https://git-lfs.github.com/) before installing

```bash
git clone https://github.com/fractalego/bert_ner.git
cd bert_ner
virtualenv .env --python=/usr/bin/python3
pip install .
```


## Example
An example can be found in the file [predict.py](bert_ner/predict.py)

```python
import os

from bert_ner.tagger import NERTagger

_path = os.path.dirname(__file__)

_filename = os.path.join(_path, '../data/ontonotes.model')

_text = "John Stephen Smith is working for Acme at the Gherkin in London."

if __name__ == '__main__':
    tagger = NERTagger(_filename)
    predictions = tagger.tag(_text)
    print(predictions)
```

With output
```python
[('john stephen smith', 'PERSON', (0, 18)), ('acme', 'ORG', (34, 38)), ('gurkin', 'FAC', (46, 52)), ('london', 'GPE', (56, 62))]
```

## Dataset
The Ontonotes 5 dataset can be downloaded from the [LDC website](https://catalog.ldc.upenn.edu/LDC2013T19).

## Results
I took measured the results both with and without BIOS tags. 
They don't seem to make much difference.

### NO BIOES

| Set | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| Dev | 0.842| 0.876| 0.858|
| Test| 0.842| 0.866| 0.853|


### With BIOES

| Set | Precision | Recall | F1 |
|:---:|:---:|:---:|:---:|
| Dev | 0.850| 0.874| 0.862|
| Test| 0.847| 0.859| 0.853|


## Comments
This is just a placeholder repo for code to reuse.
The results (85-86% F1) seem to be less impressive than the 89% that is reached in [Devlin et al](https://arxiv.org/abs/1810.04805).  
Please do not judge me on this.