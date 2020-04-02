import torch
from transformers import BertModel, BertTokenizer

from bert_ner.aux import bioes_classes, clean_tuples
from bert_ner.aux import create_data_from_sentences, batchify_sentences
from bert_ner.model import NERModel


def clean_labels(output_labels):
    return [item[2:] if item not in ['OTHER', '[CLS]'] else item for item in output_labels]


def join_tokens(words, labels):
    new_words, new_labels = [], []
    for word, label in zip(words, labels):
        if word[:2] != '##':
            new_words.append(word)
            new_labels.append(label)
            continue

        new_words[-1] += word[2:]

    return new_words, new_labels


class NERTagger:
    _MODEL = (BertModel, BertTokenizer, 'bert-base-uncased')
    _model_class, _tokenizer_class, _pretrained_weights = _MODEL
    _tokenizer = _tokenizer_class.from_pretrained(_pretrained_weights)
    _language_model = _model_class.from_pretrained(_pretrained_weights)

    def __init__(self, filename, num_batches=100):
        self._num_batches = num_batches
        self._model = NERModel(self._language_model, nout=len(bioes_classes))
        checkpoint = torch.load(filename)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._model.cuda()

    def tag(self, text):
        return self.tag_list([text])[0]

    def tag_list(self, text_list):
        sentences = create_data_from_sentences(text_list, self._tokenizer)
        batches = batchify_sentences(sentences, 10)

        entity_list = []
        for batch in batches:
            entity_list += self._predict(batch)

        return entity_list

    def _predict(self, batch):
        outputs = self._model(batch.cuda())

        out_list = []
        for inp, output in zip(batch, outputs):
            words = [self._tokenizer.convert_ids_to_tokens([i])[0] for i in list(inp)[1:]]
            output_labels = [torch.argmax(vector) for vector in output[1:]]
            output_labels = [bioes_classes[int(l)] for l in output_labels]
            output_labels = clean_labels(output_labels)
            words, output_labels = join_tokens(words, output_labels)
            predicted_tuples = clean_tuples(words, output_labels)
            predicted_tuples = [item for item in predicted_tuples if item[1] != 'OTHER']
            out_list.append(predicted_tuples)

        return out_list
