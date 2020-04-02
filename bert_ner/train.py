import os
import random
import sys

import torch
from torch import nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

from bert_ner.aux import get_data_from_tuples, get_all_sentences, batchify, test, bioes_classes
from bert_ner.model import NERModel

_path = os.path.dirname(__file__)
_train_filename = os.path.join(_path, '../data/train.conll')
_dev_filename = os.path.join(_path, '../data/dev.conll')
_save_filename = os.path.join(_path, '../data/save')

MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')


def train(train_model, batches, optimizer, criterion):
    total_loss = 0.
    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        inputs, targets = batch[0], batch[1]
        optimizer.zero_grad()
        outputs = train_model(inputs.cuda())
        loss = criterion(outputs, targets.cuda().float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    return total_loss


if __name__ == '__main__':
    model_class, tokenizer_class, pretrained_weights = MODEL
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    language_model = model_class.from_pretrained(pretrained_weights)

    print('Loading training data')
    sentences = get_all_sentences(_train_filename, max_lines=-1)
    train_data = get_data_from_tuples(sentences, tokenizer)
    train_batches = batchify(train_data, 10)

    print('Loading dev data')
    sentences = get_all_sentences(_dev_filename, max_lines=-1)
    dev_data = get_data_from_tuples(sentences, tokenizer)
    dev_batches = batchify(dev_data, 10)

    train_model = NERModel(language_model, nout=len(bioes_classes))
    train_model.cuda()

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-5)

    best_epoch = 0
    best_value = -1
    for epoch in range(20):
        random.shuffle(train_batches)
        train_model.train()
        loss = train(train_model, train_batches, optimizer, criterion)
        print('Epoch:', epoch, 'Loss:', loss)

        train_model.eval()
        score = test(train_model, dev_batches, tokenizer)

        if score > best_value:
            best_epoch = epoch
            best_value = score

        torch.save({
            'epoch': epoch,
            'model_state_dict': train_model.state_dict()},
            _save_filename + str(epoch))

        sys.stdout.flush()

    print('BEST EPOCH: ', best_epoch)
