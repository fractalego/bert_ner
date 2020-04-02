import os

import networkx as nx
import numpy as np
import torch
from networkx import DiGraph

_path = os.path.dirname(__file__)
_data_filename = os.path.join(_path, '../data/en_gum-ud-dev.conllu')
dep_edges = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'aux:pass', 'case', 'cc', 'cc:preconj',
             'ccomp', 'compound', 'compound:prt', 'conj', 'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'det:predet',
             'discourse',
             'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nmod:npmod',
             'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubj:pass', 'nummod', 'obj', 'obl', 'obl:npmod', 'obl:tmod',
             'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
dep_edges += ['join', 'cls']

initial_dep_edges = ['nsubj', 'nsubj:pass', 'iobj', 'obj', 'obl', 'obl:npmod', 'obl:tmod', 'root', 'punct', 'cls']

pos_tags = ["''", ',', '-LRB-', '-LSB-', '-RRB-', '-RSB-', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR',
            'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
            'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
pos_tags += ['JOIN', 'CLS']


def divide_lines_into_sentences(lines):
    lines = [item for item in lines if item and item[0] != '#']
    sentences = []
    sentence = []
    for line in lines:
        line = line[:-1]
        if not line:
            sentences.append(sentence)
            sentence = []
            continue
        sentence.append(line)
    return sentences


def get_relevant_tuple_from_sentences(sentences):
    tuples = []
    for sentence in sentences:
        tuple = []
        for index, line in enumerate(sentence):
            items = line.split('\t')
            tuple.append((items[1], index, int(items[6]) - 1, items[7], items[4]))
        tuples.append(tuple)

    return tuples


def create_graph_from_tuples(sentence):
    g = DiGraph()
    g.add_node(str(-1), **{'label': 'ROOT', 'name': str(-1)})

    for t in sentence:
        word = t[0]
        index = t[1]
        g.add_node(str(index), **{'label': word, 'name': str(index)})

    for index, t in enumerate(sentence):
        link_to = t[1]
        link_from = t[2]
        edge = t[3]
        g.add_edge(str(link_from), str(link_to), **{'label': edge})

    return g


def get_graphs_from_sentence_tuples(sentences):
    tuples = get_relevant_tuple_from_sentences(sentences)

    all_graphs = []
    for sentence in tuples:
        all_graphs.append(create_graph_from_tuples(sentence))

    return all_graphs


def get_sentences_from_sentence_tuples(sentence_tuples):
    all_sentences = []
    for tuple in sentence_tuples:
        sentence = ''
        for item in tuple:
            sentence += item[0] + ' '
        all_sentences.append(sentence[:-1])

    return all_sentences


def get_new_sentence_tuples(sentence_tuples, tokenizer):
    new_tuples = []
    for tuples in sentence_tuples:
        new_tuple = []
        new_index = len(tuples)
        for tuple in tuples:
            word = tuple[0]
            new_tokens = tokenizer.tokenize(word)
            if len(new_tokens) == 1:
                new_tuple.append((tuple[0], tuple[1], tuple[2], tuple[3], tuple[4]))
                continue
            new_tuple.append((new_tokens[0], tuple[1], tuple[2], tuple[3], tuple[4]))
            old_index = tuple[1]
            for new_token in new_tokens[1:]:
                word = new_token
                link_to = new_index
                link_from = old_index
                new_tuple.append((word, link_to, link_from, 'join', 'JOIN'))
                old_index = new_index
                new_index += 1

        new_tuples.append(new_tuple)

    return new_tuples


def create_data(filename, tokenizer, limit=None):
    sentence_tuples = get_relevant_tuple_from_sentences(divide_lines_into_sentences(open(filename).readlines()))
    all_sentences = get_sentences_from_sentence_tuples(sentence_tuples)
    sentence_tuples = get_new_sentence_tuples(sentence_tuples, tokenizer)

    data = []

    if limit:
        all_sentences = all_sentences[:limit]
    for tuples, sentence in zip(sentence_tuples, all_sentences):
        input_ids = torch.tensor([[101] + tokenizer.encode(sentence, add_special_tokens=False)])
        one_hot_edges = [create_one_hot_vector(dep_edges.index('cls'), len(dep_edges))] \
                        + [create_one_hot_vector(dep_edges.index(item[3]), len(dep_edges)) for item in tuples]
        edges = torch.tensor(np.array(one_hot_edges))
        one_hot_pos = [create_one_hot_vector(pos_tags.index('CLS'), len(pos_tags))] \
                      + [create_one_hot_vector(pos_tags.index(item[4]), len(pos_tags)) for item in tuples]
        pos = torch.tensor(np.array(one_hot_pos))
        with torch.no_grad():
            graph = create_graph_from_tuples(tuples)
            adjacency_matrix = torch.tensor(np.array(nx.adjacency_matrix(graph).toarray()).transpose([1, 0]))
            data.append((input_ids, edges, adjacency_matrix, pos))

    return data


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


def test(eval_model, batches):
    edges_total = 0
    edges_tp = 0
    adj_total = 0
    adj_tp = 0
    las_total = 0
    las_tp = 0
    pos_total = 0
    pos_tp = 0
    for i, batch in enumerate(batches):
        inputs, edges_targets, adj_targets, pos_targets = batch[0], batch[1], batch[2], batch[3]
        edges_outputs, adj_outputs, pos_outputs = eval_model(inputs.cuda())

        for target, output, pos in zip(edges_targets, edges_outputs, pos_targets):
            # skipping the first vector because it is always the same CLS
            pos_labels = [torch.argmax(vector) for vector in pos[1:]]
            target_labels = [get_edge_label_from_vector(vector) for vector in target[1:]]
            output_labels = [get_edge_label_from_vector(vector) for vector in output[1:]]

            join_pos = int(sum([t == pos_tags.index('JOIN') for t in pos_labels]))

            edges_tp += sum([t == o for t, o in zip(target_labels, output_labels)]) - join_pos
            edges_total += len(target_labels) - join_pos

        for target, output, pos in zip(adj_targets, adj_outputs, pos_targets):
            # skipping the first vector because it has no head

            pos_labels = [torch.argmax(vector) for vector in pos[1:]]
            target_labels = [torch.argmax(vector) for vector in target[1:]]
            output_labels = [torch.argmax(vector) for vector in output[1:]]

            join_pos = int(sum([t == pos_tags.index('JOIN') for t in pos_labels]))

            adj_tp += int(sum([t == o for t, o in zip(target_labels, output_labels)])) - join_pos
            adj_total += len(target_labels) - join_pos

        for target, output, edge_target, edge_output, pos in zip(adj_targets, adj_outputs,
                                                                 edges_targets, edges_outputs,
                                                                 pos_targets):
            # skipping the first vector because it has no head

            pos_labels = [torch.argmax(vector) for vector in pos[1:]]
            target_labels = [torch.argmax(vector) for vector in target[1:]]
            output_labels = [torch.argmax(vector) for vector in output[1:]]
            target_edges = [torch.argmax(vector) for vector in edge_target[1:]]
            output_edges = [torch.argmax(vector) for vector in edge_output[1:]]

            join_pos = int(sum([t == pos_tags.index('JOIN') for t in pos_labels]))

            las_tp += int(sum([t == o and et == eo
                               for t, o, et, eo in zip(target_labels, output_labels, target_edges, output_edges)])) \
                      - join_pos
            las_total += len(target_labels) - join_pos

        for target, output in zip(pos_targets, pos_outputs):
            # skipping the first vector because it is always the same CLS
            target_labels = [torch.argmax(vector) for vector in target[1:]]
            output_labels = [torch.argmax(vector) for vector in output[1:]]
            join_pos = int(sum([t == pos_tags.index('JOIN') for t in target_labels]))

            pos_tp += int(sum([t == o for t, o in zip(target_labels, output_labels)])) - join_pos
            pos_total += len(target_labels) - join_pos

    LAS_score = las_tp / las_total
    UAS_score = adj_tp / adj_total
    edges_score = edges_tp / edges_total
    print('   Edges tp:', edges_score)
    print('   UAS tp:', UAS_score)
    print('   LAS tp:', LAS_score)
    print('   POS tp:', pos_tp / pos_total)

    return LAS_score, UAS_score, edges_score


def get_edge_label_from_vector(vector):
    index = torch.argmax(vector)
    return dep_edges[index]


def create_graph(edges, adj, pos, tokens, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(tokens[1:])

    words = []
    joined_word = ''
    for token in tokens[::-1]:
        words.append(token + joined_word)
        if token[:2] == '##':
            joined_word = token[2:] + joined_word
        else:
            joined_word = ''
    words = words[::-1]

    pos_indices = [torch.argmax(vector) for vector in pos[1:]]
    edges_indices = [torch.argmax(vector) for vector in edges[1:]]
    adj_indices = [int(torch.argmax(vector)) - 1 for vector in adj][1:]

    pos_labels = [pos_tags[index] for index in pos_indices]
    edge_labels = [dep_edges[index] for index in edges_indices]

    g = DiGraph()
    for index, item in enumerate(zip(pos_labels, edge_labels, words, adj_indices)):
        pos = item[0]
        edge = item[1]
        token = item[2]
        link_from = item[3]

        if pos == 'JOIN':
            continue

        g.add_node(str(index) + '_' + token, **{'pos': pos, 'token': token})
        if link_from != -1:
            g.add_edge(str(index) + '_' + token,
                       str(link_from) + '_' + words[link_from],
                       **{'label': edge})

    return g
