import numpy as np
import torch
from tqdm import tqdm

classes = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL",
           "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]
bclasses = ['B-' + item for item in classes]
iclasses = ['I-' + item for item in classes]
eclasses = ['E-' + item for item in classes]
sclasses = ['S-' + item for item in classes]
bioes_classes = bclasses + iclasses + eclasses + sclasses

bioes_classes += ['OTHER', 'CLS']


def get_all_sentences(filename, max_lines=-1):
    file = open(filename)
    sentences = []
    items = []
    old_entity = 'OTHER'
    for line in tqdm(file.readlines()[:max_lines]):
        if line[0] == '#':
            continue
        elements = line.split()
        if len(elements) < 5:
            if items != []:
                sentences.append(items)
            items = []
            continue
        word = elements[3].strip()
        tag = elements[4].strip()
        entity, old_entity = decide_entity(elements[10].strip(), old_entity)
        items.append((word, tag, entity))
    return sentences


def decide_entity(string, prior_entity):
    if string == '*)':
        return prior_entity, 'OTHER'
    if string == '*':
        return prior_entity, prior_entity
    entity = 'OTHER'
    for item in classes:
        if string.find(item) != -1:
            entity = item
    prior_entity = 'OTHER'
    if string.find(')') == -1:
        prior_entity = entity
    return entity, prior_entity


def add_bioes(tags):
    bioes_tags = []

    for i in range(len(tags)):
        curr = tags[i]
        if curr in ['OTHER', '[CLS]']:
            bioes_tags.append(curr)
            continue

        prior = tags[i - 1] if i - 1 >= 0 else 'OTHER'
        nxt = tags[i + 1] if i + 1 < len(tags) else 'OTHER'

        if prior != curr and nxt != curr:
            bioes_tags.append('S-' + curr)
        if prior == curr and nxt == curr:
            bioes_tags.append('I-' + curr)
        if prior != curr and nxt == curr:
            bioes_tags.append('B-' + curr)
        if prior == curr and nxt != curr:
            bioes_tags.append('E-' + curr)

    return bioes_tags


def get_sentences_and_targets_from_sentence_tuples(tuples_list):
    all_sentences = []
    all_targets = []
    for tuple in tuples_list:
        sentence = ''
        for item in tuple:
            sentence += item[0] + ' '
        all_sentences.append(sentence[:-1])
        all_targets.append(add_bioes([item[2] for item in tuple]))
    return all_sentences, all_targets


def create_one_hot_vector(index, lenght):
    vector = [0.] * lenght
    vector[index] = 1.
    return vector


def get_new_targets(sentence, targets, tokenizer):
    new_targets = []
    for word, target in zip(sentence.split(), targets):
        new_tokens = tokenizer.tokenize(word)
        if len(new_tokens) == 1:
            new_targets.append(target)
            continue
        new_targets.append(target)
        for _ in new_tokens[1:]:
            new_targets.append(target)
    return new_targets


def get_data_from_tuples(tuples_list, tokenizer):
    all_data = []
    sentences, targets = get_sentences_and_targets_from_sentence_tuples(tuples_list)
    for sentence, target in tqdm(zip(sentences, targets), total=len(sentences)):
        input_ids = torch.tensor([[101] + tokenizer.encode(sentence, add_special_tokens=False)])
        target = get_new_targets(sentence, target, tokenizer)
        one_hot_labels = [create_one_hot_vector(bioes_classes.index('CLS'), len(bioes_classes))] \
                         + [create_one_hot_vector(bioes_classes.index(item), len(bioes_classes)) for item in target]
        if len(one_hot_labels) != input_ids.shape[1]:
            print('Error creating data from sentence:', sentence)

        labels = torch.tensor(np.array(one_hot_labels))
        all_data.append((input_ids, labels))

    return all_data


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def batchify(data, n):
    len_dict = {}
    for item in data:
        length = item[0].shape[1]
        try:
            len_dict[length].append(item)
        except:
            len_dict[length] = [item]

    batch_chunks = []
    for k in len_dict.keys():
        vectors = len_dict[k]
        batch_chunks += chunks(vectors, n)

    batches = []
    for chunk in batch_chunks:
        input = torch.stack([item[0][0] for item in chunk])
        labels = torch.stack([item[1] for item in chunk])
        batches.append((input, labels))

    return batches


def erase_non_entities(all_words, all_entities, all_idx):
    return [(w, e, i) for w, e, i in zip(all_words, all_entities, all_idx) if e and w != ' ']


def join_consecutive_tuples(tuples):
    for i in range(len(tuples) - 1):
        curr_type = tuples[i][1]
        curr_end_idx = tuples[i][2][1]
        next_type = tuples[i + 1][1]
        next_start_idx = tuples[i + 1][2][0]
        if curr_type == next_type and curr_end_idx == next_start_idx - 1:
            curr_word = tuples[i][0]
            next_word = tuples[i + 1][0]
            curr_start_idx = tuples[i][2][0]
            next_end_idx = tuples[i + 1][2][1]
            tuples[i + 1] = (curr_word + ' ' + next_word,
                             curr_type,
                             (curr_start_idx, next_end_idx))
            tuples[i] = ()
    tuples = [t for t in tuples if t]
    return tuples


def clean_tuples(all_words, all_entities):
    all_idx = []
    index = 0
    for word in all_words:
        all_idx.append((index, index + len(word)))
        index += len(word) + 1
    tuples = erase_non_entities(all_words, all_entities, all_idx)
    tuples = join_consecutive_tuples(tuples)
    return tuples


def test(eval_model, batches, tokenizer):
    total = 0
    tp = 0
    precision = 0
    recall = 0
    total_predicted = 0
    total_target = 0
    for i, batch in tqdm(enumerate(batches), total=len(batches)):
        inputs, targets = batch[0], batch[1]
        outputs = eval_model(inputs.cuda())

        for input, target, output in zip(inputs, targets, outputs):
            # skipping the first vector because it is [CLS]

            target_labels = [torch.argmax(vector) for vector in target[1:]]
            target_labels = [bioes_classes[int(l)] for l in target_labels]
            output_labels = [torch.argmax(vector) for vector in output[1:]]
            output_labels = [bioes_classes[int(l)] for l in output_labels]

            words = [tokenizer.convert_ids_to_tokens([i])[0] for i in list(input)[1:]]
            predicted_tuples = clean_tuples(words, output_labels)
            target_tuples = clean_tuples(words, target_labels)
            predicted_tuples = [item for item in predicted_tuples if item[1] != 'OTHER']
            target_tuples = [item for item in target_tuples if item[1] != 'OTHER']

            tp += int(sum([t == o for t, o in zip(target_labels, output_labels)]))
            total += len(target_labels)

            if predicted_tuples:
                precision += len(set(predicted_tuples) & set(target_tuples))
                total_predicted += len(predicted_tuples)
            if target_tuples:
                recall += len(set(predicted_tuples) & set(target_tuples))
                total_target += len(target_tuples)

    score = tp / total
    if total_target:
        recall /= total_target
    if total_predicted:
        precision /= total_predicted
    print('   Words tp:', score)
    print('   Precision :', precision)
    print('   Recall :', recall)
    if precision + recall:
        print('   F1 :', 2 * precision * recall / (precision + recall))

    return score


def create_data_from_sentences(all_sentences, tokenizer, limit=None):
    data = []

    if limit:
        all_sentences = all_sentences[:limit]
    for sentence in all_sentences:
        sentence = sentence.replace('\n', '')
        input_ids = torch.tensor([[101] + tokenizer.encode(sentence, add_special_tokens=False)])
        with torch.no_grad():
            data.append(input_ids)

    return data


def batchify_sentences(data, n):
    len_dict = {}
    for item in data:
        length = item.shape[1]
        try:
            len_dict[length].append(item)
        except:
            len_dict[length] = [item]

    batch_chunks = []
    for k in len_dict.keys():
        vectors = len_dict[k]
        for batch in chunks(vectors, n):
            batch_chunks.append(batch[0])

    return batch_chunks
