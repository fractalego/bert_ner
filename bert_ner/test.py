import os

import torch
from transformers import BertModel, BertTokenizer

from bert_ner.aux import get_data_from_tuples, get_all_sentences, batchify, test, bioes_classes
from bert_ner.model import NERModel

_path = os.path.dirname(__file__)
_test_filename = os.path.join(_path, '../data/test.conll')
_save_filename = os.path.join(_path, '../data/ontonotes.model')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')

if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    print('Loading test data')
    sentences = get_all_sentences(_test_filename, max_lines=-1)
    test_data = get_data_from_tuples(sentences, tokenizer)
    test_batches = batchify(test_data, 10)

    model = NERModel(language_model, nout=len(bioes_classes))
    checkpoint = torch.load(_save_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cuda()
    model.eval()
    test(model, test_batches, tokenizer)
