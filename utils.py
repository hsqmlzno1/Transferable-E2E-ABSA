import string
from nltk import ngrams
import numpy as np
# DO NOT change the random seed, otherwise, the train-test split will be inconsistent with those in the baselines
np.random.seed(7894)
import os
import pickle
from evals import *
import math

def ot2bio_ote(ote_tag_sequence):
    """
    ot2bio function for ote tag sequence
    :param ote_tag_sequence:
    :return:
    """
    new_ote_sequence = []
    n_tag = len(ote_tag_sequence)
    prev_ote_tag = '$$$'
    for i in range(n_tag):
        cur_ote_tag = ote_tag_sequence[i]
        assert cur_ote_tag == 'O' or cur_ote_tag == 'T'
        if cur_ote_tag == 'O':
            new_ote_sequence.append(cur_ote_tag)
        else:
            # cur_ote_tag is T
            if prev_ote_tag == 'T':
                new_ote_sequence.append('I')
            else:
                # cur tag is at the beginning of the opinion target
                new_ote_sequence.append('B')
        prev_ote_tag = cur_ote_tag
    return new_ote_sequence


def ot2bio_ts(ts_tag_sequence):
    """
    ot2bio function for ts tag sequence
    :param ts_tag_sequence:
    :return:
    """
    new_ts_sequence = []
    n_tag = len(ts_tag_sequence)
    prev_pos = '$$$'
    for i in range(n_tag):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            # current tag is subjective tag, i.e., cur_pos is T
            # print(cur_ts_tag)
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            if cur_pos == prev_pos:
                # prev_pos is T
                new_ts_sequence.append('I-%s' % cur_sentiment)
            else:
                # prev_pos is O
                new_ts_sequence.append('B-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bio(ote_tag_sequence, ts_tag_sequence):
    """
    perform ot--->bio for both ote tag sequence and ts tag sequence
    :param ote_tag_sequence: input tag sequence of opinion target extraction
    :param ts_tag_sequence: input tag sequence of targeted sentiment
    :return:
    """
    new_ote_sequence = ot2bio_ote(ote_tag_sequence=ote_tag_sequence)
    new_ts_sequence = ot2bio_ts(ts_tag_sequence=ts_tag_sequence)
    assert len(new_ts_sequence) == len(ts_tag_sequence)
    assert len(new_ote_sequence) == len(ote_tag_sequence)
    return new_ote_sequence, new_ts_sequence


def ot2bio_ote_batch(ote_tag_seqs):
    """
    batch version of function ot2bio_ote
    :param ote_tags:
    :return:
    """
    new_ote_tag_seqs = []
    n_seqs = len(ote_tag_seqs)
    for i in range(n_seqs):
        new_ote_seq = ot2bio_ote(ote_tag_sequence=ote_tag_seqs[i])
        new_ote_tag_seqs.append(new_ote_seq)
    return new_ote_tag_seqs


def ot2bio_ts_batch(ts_tag_seqs):
    """
    batch version of function ot2bio_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bio_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def ot2bio_batch(ote_tags, ts_tags):
    """
    batch version of function ot2bio
    :param ote_tags: a batch of ote tag sequence
    :param ts_tags: a batch of ts tag sequence
    :return:
    """
    new_ote_tags, new_ts_tags = [], []
    assert len(ote_tags) == len(ts_tags)
    n_seqs = len(ote_tags)
    for i in range(n_seqs):
        ote, ts = ot2bio(ote_tag_sequence=ote_tags[i], ts_tag_sequence=ts_tags[i])
        new_ote_tags.append(ote)
        new_ts_tags.append(ts)
    return new_ote_tags, new_ts_tags


def ot2bieos_ote(ote_tag_sequence):
    """
    ot2bieos function for ote task
    :param ote_tag_sequence:
    :return:
    """
    n_tags = len(ote_tag_sequence)
    new_ote_sequence = []
    prev_ote_tag = '$$$'
    for i in range(n_tags):
        cur_ote_tag = ote_tag_sequence[i]
        if cur_ote_tag == 'O':
            new_ote_sequence.append('O')
        else:
            # cur_ote_tag is T
            if prev_ote_tag != cur_ote_tag:
                # prev_ote_tag is O, new_cur_tag can only be B or S
                if i == n_tags - 1:
                    new_ote_sequence.append('S')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    new_ote_sequence.append('B')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    new_ote_sequence.append('S')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
            else:
                # prev_tag is T, new_cur_tag can only be I or E
                if i == n_tags - 1:
                    new_ote_sequence.append('E')
                elif ote_tag_sequence[i + 1] == cur_ote_tag:
                    # next_tag is T
                    new_ote_sequence.append('I')
                elif ote_tag_sequence[i + 1] != cur_ote_tag:
                    # next_tag is O
                    new_ote_sequence.append('E')
                else:
                    raise Exception("Invalid ner tag value: %s" % cur_ote_tag)
        prev_ote_tag = cur_ote_tag
    return new_ote_sequence


def ot2bieos_ts(ts_tag_sequence):
    """
    ot2bieos function for ts task
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    n_tags = len(ts_tag_sequence)
    new_ts_sequence = []
    prev_pos = '$$$'
    for i in range(n_tags):
        cur_ts_tag = ts_tag_sequence[i]
        if cur_ts_tag == 'O':
            new_ts_sequence.append('O')
            cur_pos = 'O'
        else:
            cur_pos, cur_sentiment = cur_ts_tag.split('-')
            # cur_pos is T
            if cur_pos != prev_pos:
                # prev_pos is O and new_cur_pos can only be B or S
                if i == n_tags - 1:
                    new_ts_sequence.append('S-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('S-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('B-%s' % cur_sentiment)
            else:
                # prev_pos is T and new_cur_pos can only be I or E
                if i == n_tags - 1:
                    new_ts_sequence.append('E-%s' % cur_sentiment)
                else:
                    next_ts_tag = ts_tag_sequence[i + 1]
                    if next_ts_tag == 'O':
                        new_ts_sequence.append('E-%s' % cur_sentiment)
                    else:
                        new_ts_sequence.append('I-%s' % cur_sentiment)
        prev_pos = cur_pos
    return new_ts_sequence


def ot2bieos(ote_tag_sequence, ts_tag_sequence):
    """
    perform ot-->bieos for both ote tag and ts tag sequence
    :param ote_tag_sequence: input tag sequence of opinion target extraction
    :param ts_tag_sequence: input tag sequence of targeted sentiment
    :return:
    """
    # new tag sequences of opinion target extraction and targeted sentiment
    new_ote_sequence = ot2bieos_ote(ote_tag_sequence=ote_tag_sequence)
    new_ts_sequence = ot2bieos_ts(ts_tag_sequence=ts_tag_sequence)
    assert len(ote_tag_sequence) == len(new_ote_sequence)
    assert len(ts_tag_sequence) == len(new_ts_sequence)
    return new_ote_sequence, new_ts_sequence


def ot2bieos_ote_batch(ote_tag_seqs):
    """
    batch version of function ot2bieos_ote
    :param ote_tags:
    :return:
    """
    new_ote_tag_seqs = []
    n_seqs = len(ote_tag_seqs)
    for i in range(n_seqs):
        new_ote_seq = ot2bieos_ote(ote_tag_sequence=ote_tag_seqs[i])
        new_ote_tag_seqs.append(new_ote_seq)
    return new_ote_tag_seqs


def ot2bieos_ts_batch(ts_tag_seqs):
    """
    batch version of function ot2bieos_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = ot2bieos_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def ot2bieos_batch(ote_tags, ts_tags):
    """
    batch version of function ot2bieos
    :param ote_tags: a batch of ote tag sequence
    :param ts_tags: a batch of ts tag sequence
    :return:
    :param ote_tags:
    :param ts_tags:
    :return:
    """
    new_ote_tags, new_ts_tags = [], []
    assert len(ote_tags) == len(ts_tags)
    n_seqs = len(ote_tags)
    for i in range(n_seqs):
        ote, ts = ot2bieos(ote_tag_sequence=ote_tags[i], ts_tag_sequence=ts_tags[i])
        new_ote_tags.append(ote)
        new_ts_tags.append(ts)
    return new_ote_tags, new_ts_tags


def bio2ot_ote(ote_tag_sequence):
    """
    perform bio-->ot for ote tag sequence
    :param ote_tag_sequence:
    :return:
    """
    new_ote_sequence = []
    n_tags = len(ote_tag_sequence)
    for i in range(n_tags):
        ote_tag = ote_tag_sequence[i]
        if ote_tag == 'B' or ote_tag == 'I':
            new_ote_sequence.append('T')
        else:
            new_ote_sequence.append('I')
    return new_ote_sequence


def bio2ot_ts(ts_tag_sequence):
    """
    perform bio-->ot for ts tag sequence
    :param ts_tag_sequence:
    :return:
    """
    new_ts_sequence = []
    n_tags = len(ts_tag_sequence)
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        if ts_tag == 'O':
            new_ts_sequence.append('O')
        else:
            pos, sentiment = ts_tag.split('-')
            new_ts_sequence.append('T-%s' % sentiment)
    return new_ts_sequence


def bio2ot(ote_tag_sequence, ts_tag_sequence):
    """
    perform bio-->ot for both ote and ts tag sequence
    :param ote_tag_sequence: tag sequence for opinion target extraction
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return:
    """
    assert len(ote_tag_sequence) == len(ts_tag_sequence)
    new_ote_sequence = bio2ot_ote(ote_tag_sequence=ote_tag_sequence)
    new_ts_sequence = bio2ot_ts(ts_tag_sequence=ts_tag_sequence)
    assert len(new_ote_sequence) == len(ote_tag_sequence)
    assert len(new_ts_sequence) == len(ts_tag_sequence)
    return new_ote_sequence, new_ts_sequence


def bio2ot_ote_batch(ote_tag_seqs):
    """
    batch version of function bio2ot_ote
    :param ote_tag_seqs: ote tag sequences
    :return:
    """
    new_ote_tag_seqs = []
    n_seqs = len(ote_tag_seqs)
    for i in range(n_seqs):
        new_ote_seq = bio2ot_ote(ote_tag_sequence=ote_tag_seqs[i])
        new_ote_tag_seqs.append(new_ote_seq)
    return new_ote_tag_seqs


def bio2ot_ts_batch(ts_tag_seqs):
    """
    batch version of function bio2ot_ts
    :param ts_tag_seqs:
    :return:
    """
    new_ts_tag_seqs = []
    n_seqs = len(ts_tag_seqs)
    for i in range(n_seqs):
        new_ts_seq = bio2ot_ts(ts_tag_sequence=ts_tag_seqs[i])
        new_ts_tag_seqs.append(new_ts_seq)
    return new_ts_tag_seqs


def bio2ot_batch(ote_tags, ts_tags):
    """
    batch version of function bio2ot
    :param ote_tags: a batch of ote tag sequence
    :param ts_tags: a batch of ts tag sequence
    :return:
    """
    new_ote_tags, new_ts_tags = [], []
    assert len(ote_tags) == len(ts_tags)
    n_seqs = len(ote_tags)
    for i in range(n_seqs):
        ote, ts = bio2ot(ote_tag_sequence=ote_tags[i], ts_tag_sequence=ts_tags[i])
        new_ote_tags.append(ote)
        new_ts_tags.append(ts)
    return new_ote_tags, new_ts_tags


# TODO
def bieos2ot(tag_sequence):
    """
    transform BIEOS tag sequence to OT tag sequence
    :param tag_sequence: input tag sequence
    :return:
    """
    new_sequence = []
    for t in tag_sequence:
        assert t == 'B' or t == 'I' or t == 'E' or t == 'O' or t == 'S'
        if t == 'O':
            new_sequence.append(t)
        else:
            new_sequence.append('T')
    assert len(new_sequence) == len(tag_sequence)
    return new_sequence


def get_vocab(data_set):
    """
    build the vocabulary of the whole dataset
    :param train_set:
    :param test_set:
    :return:
    """
    vocab = {'PADDING': 0, 'PUNCT': 1}
    inv_vocab = {0: 'PADDING', 1: 'PUNCT'}
    wid = 2
    max_len = -1
    for record in data_set:
        assert 'words' in record
        words = record['words']
        if len(words) > max_len:
            max_len = len(words)
        for w in words:
            if w not in vocab:
                vocab[w] = wid
                inv_vocab[wid] = w
                wid += 1
    print("The maximum length of the sentence is %d" % max_len)
    print("Find %s different words in the dataset" % len(vocab))
    char_string = ''
    for w in vocab:
        char_string += w
    chars = list(set(char_string))
    cid, char_vocab = 0, {}
    for ch in chars:
        if ch not in char_vocab:
            char_vocab[ch] = cid
            cid += 1
    print("Find %s different chars in the dataset" % len(char_vocab))
    return vocab, char_vocab, max_len


def read_lexicon():
    """
    read sentiment lexicon from the disk
    :return:
    """
    # path = 'mpqa_full.txt'

    path_list = ['mpqa_full.txt', 'sent_lexicon_14semeval_laptop.txt',
            'sent_lexicon_15semeval_rest.txt',
            'sent_lexicon_16semeval_rest.txt']
    sent_lexicon = {}
    for path in path_list:
        with open('work/lexicon/'+path) as fp:
            for line in fp:
                word, polarity = line.strip().split('\t')
                if word not in sent_lexicon:
                    sent_lexicon[word] = int(polarity)
        fp.close()

    return sent_lexicon


def read_data(path):
    """
    read data from the specified path
    :param path: path of dataset
    :return:
    """
    dataset = []
    with open(path) as fp:
        for line in fp:
            record = {}
            # print(line.strip())
            # print(line.strip().split('####'))
            sent, tag_string = line.strip().split('####')

            record['sentence'] = sent
            word_tag_pairs = tag_string.split(' ')
            # tag sequence for targeted sentiment
            ts_tags = []
            # tag sequence for opinion target extraction
            ote_tags = []
            # word sequence
            words = []
            for item in word_tag_pairs:
                # valid label is: O, T-POS, T-NEG, T-NEU
                eles = item.split('=')
                if len(eles) == 2:
                    word, tag = eles
                elif len(eles) > 2:
                    tag = eles[-1]
                    word = (len(eles) - 2) * "="
                if word not in string.punctuation:
                    # lowercase the words
                    words.append(word.lower())
                else:
                    # replace punctuations with a special token
                    words.append('PUNCT')
                if tag == 'O':
                    ote_tags.append('O')
                    ts_tags.append('O')
                elif tag == 'T-POS':
                    ote_tags.append('T')
                    ts_tags.append('T-POS')
                elif tag == 'T-NEG':
                    ote_tags.append('T')
                    ts_tags.append('T-NEG')
                elif tag == 'T-NEU':
                    ote_tags.append('T')
                    ts_tags.append('T-NEU')
                else:
                    raise Exception('Invalid tag %s!!!' % tag)
            flag = False
            for ote_tag in ote_tags:
                if ote_tag != 'O':
                    flag = True
                    break
            # flag = True
            if flag:
                record['words'] = list(words)
                record['ote_raw_tags'] = list(ote_tags)
                record['ts_raw_tags'] = list(ts_tags)
                dataset.append(record)
    print("Obtain %s records from %s" % (len(dataset), path))
    return dataset


def set_wid(dataset, vocab, win=1):
    """
    set wid field for the dataset
    :param dataset: dataset
    :param vocab: vocabulary
    :param win: context window size, for window-based input, should be an odd number
    :return: dataset with field wid
    """
    n_records = len(dataset)
    for i in range(n_records):
        words = dataset[i]['words']
        lm_labels = []
        # set labels for the auxiliary language modeling task
        for w in words:
            lm_labels.append(vocab[w])
        dataset[i]['lm_labels'] = list(lm_labels)
        n_padded_words = win // 2
        pad_left = ['PADDING' for _ in range(n_padded_words)]
        pad_right = ['PADDING' for _ in range(n_padded_words)]
        padded_words = pad_left + words + pad_right
        # the window-based input
        win_input = list(ngrams(padded_words, win))
        assert len(win_input) == len(words)
        n_grams = []
        for t in win_input:
            n_grams.append(t)
        wids = [[vocab[w] for w in ngram] for ngram in n_grams]
        dataset[i]['wids'] = list(wids)
    return dataset


def set_cid(dataset, char_vocab):
    """
    set cid field for the records in the dataset
    :param dataset: dataset
    :param char_vocab: vocabulary of character
    :return:
    """
    n_records = len(dataset)
    cids = []
    for i in range(n_records):
        words = dataset[i]['words']
        cids = []
        for w in words:
            cids.append([char_vocab[ch] for ch in list(w)])
        dataset[i]['cids'] = list(cids)
    return dataset


def set_labels(dataset, tagging_schema='BIO'):
    """
    set ote_label and ts_label for the dataset
    :param dataset: dataset without ote_label and ts_label fields
    :param tagging_schema: tagging schema of ote_tag and ts_tag
    :return:
    """
    if tagging_schema == 'OT':
        ote_tag_vocab = {'O': 0, 'T': 1}
        ts_tag_vocab = {'O': 0, 'T-POS': 1, 'T-NEG': 2, 'T-NEU': 3}
    elif tagging_schema == 'BIO':
        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2}
        ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'B-NEG': 3, 'I-NEG': 4,
                        'B-NEU': 5, 'I-NEU': 6}
    elif tagging_schema == 'BIEOS':
        ote_tag_vocab = {'O': 0, 'B': 1, 'I': 2, 'E': 3, 'S': 4}
        ts_tag_vocab = {'O': 0, 'B-POS': 1, 'I-POS': 2, 'E-POS': 3, 'S-POS': 4,
                        'B-NEG': 5, 'I-NEG': 6, 'E-NEG': 7, 'S-NEG': 8,
                        'B-NEU': 9, 'I-NEU': 10, 'E-NEU': 11, 'S-NEU': 12}
    else:
        raise Exception("Invalid tagging schema %s" % tagging_schema)
    n_records = len(dataset)
    for i in range(n_records):
        ote_tags = dataset[i]['ote_raw_tags']
        ts_tags = dataset[i]['ts_raw_tags']
        if tagging_schema == 'OT':
            pass
        elif tagging_schema == 'BIO':
            ote_tags, ts_tags = ot2bio(ote_tag_sequence=ote_tags, ts_tag_sequence=ts_tags)
        elif tagging_schema == 'BIEOS':
            ote_tags, ts_tags = ot2bieos(ote_tag_sequence=ote_tags, ts_tag_sequence=ts_tags)
        else:
            raise Exception("Invalid tagging schema %s" % tagging_schema)
        ote_labels = [ote_tag_vocab[t] for t in ote_tags]
        ts_labels = [ts_tag_vocab[t] for t in ts_tags]
        dataset[i]['ote_tags'] = list(ote_tags)
        dataset[i]['ts_tags'] = list(ts_tags)
        dataset[i]['ote_labels'] = list(ote_labels)
        dataset[i]['ts_labels'] = list(ts_labels)
    return dataset, ote_tag_vocab, ts_tag_vocab


def set_lm_labels(dataset, vocab, stm_lex, stm_win=3):
    """
    set labels of bi-directional language modeling and sentiment-aware language modeling
    :param dataset: dataset
    :param vocab: vocabulary
    :param stm_lex: sentiment lexicon
    :param stm_win: window size (i.e., length) of sentiment context
    :return:
    """
    n_records = len(dataset)
    for i in range(n_records):
        words = dataset[i]['words']
        # labels of language modeling and sentiment aware language modeling
        lm_labels_f, lm_labels_b = [], []
        n_w = len(words)
        # language modeling in forward direction
        for j in range(n_w):
            if j == n_w - 1:
                next_word = -1
            else:
                if words[j+1] in stm_lex:
                    next_word = stm_lex[words[j+1]]
                else:
                    next_word = -1
            next_word = -1* next_word + 1
            lm_labels_f.append(next_word)
        for j in range(n_w-1, -1, -1):
            if j == 0:
                next_word = -1
            else:
                if words[j-1] in stm_lex:
                    next_word = stm_lex[words[j-1]]
                else:
                    next_word = -1
            next_word = -1* next_word + 1
            lm_labels_b.append(next_word)
        dataset[i]['lm_labels_f'] = list(lm_labels_f)
        dataset[i]['lm_labels_b'] = list(lm_labels_b)[::-1]
        # sentiment aware language modeling
        stm_lm_labels = []
        opn_labels = []
        for j in range(n_w):
            if words[j] in stm_lex:
                opn_labels.append(1)
            else:
                opn_labels.append(0)
            # left boundary of sentimental context
            stm_ctx_lb = j - stm_win
            if stm_ctx_lb < 0:
                stm_ctx_lb = 0
            stm_ctx_rb = j + stm_win + 1
            left_ctx = words[stm_ctx_lb:j]
            right_ctx = words[j+1:stm_ctx_rb]
            stm_ctx = left_ctx + right_ctx
            flag = False
            for w in stm_ctx:
                if w in stm_lex:
                    flag = True
                    break
            if flag:
                stm_lm_labels.append(1)
            else:
                stm_lm_labels.append(0)
        dataset[i]['stm_lm_labels'] = list(stm_lm_labels)
        dataset[i]['opn_labels'] = list(opn_labels)
    return dataset


def set_padding(dataset, max_len):

    n_records = len(dataset)
    for i in range(n_records):
        sent_len = len(dataset[i]['words'])
        words = dataset[i]['words'] + ['PADDING'] * (max_len-sent_len)
        ote_tags = dataset[i]['ote_raw_tags'] + ['O'] * (max_len-sent_len)
        ts_tags = dataset[i]['ts_raw_tags'] + ['O'] * (max_len-sent_len)

        dataset[i]['words'] = list(words)
        dataset[i]['ote_raw_tags'] = list(ote_tags)
        dataset[i]['ts_raw_tags'] = list(ts_tags)
        dataset[i]['length'] = sent_len

    return dataset

def build_dataset(source_domain, target_domain, input_win=1, tagging_schema='BIO', stm_win=1):
    """
    build dataset for model training, development and inference
    :param ds_name: dataset name
    :param input_win: window size input
    :param tagging_schema: tagging schema
    :param stm_win: window size of context for the OE component
    :return:
    """
    # read mpqa sentiment lexicon
    stm_lex = read_lexicon()
    # paths of training and testing dataset
    train_path  = './data/%s_train.txt' % source_domain
    val_path    = './data/%s_test.txt'  % source_domain
    tar_un_path = './data/%s_train.txt' % target_domain
    test_path   = './data/%s_test.txt'  % target_domain
    # loaded datasets
    train_set  = read_data(path=train_path)
    val_set    = read_data(path=val_path)
    tar_un_set = read_data(path=tar_un_path)
    test_set   = read_data(path=test_path)

    vocab, char_vocab, max_len = get_vocab(train_set+val_set+tar_un_set+test_set)

    train_set  = set_padding(dataset=train_set, max_len=max_len)
    val_set    = set_padding(dataset=val_set, max_len=max_len)
    tar_un_set = set_padding(dataset=tar_un_set, max_len=max_len)
    test_set   = set_padding(dataset=test_set, max_len=max_len)

    train_set  = set_wid(dataset=train_set, vocab=vocab, win=input_win)
    val_set    = set_wid(dataset=val_set, vocab=vocab, win=input_win)
    tar_un_set = set_wid(dataset=tar_un_set, vocab=vocab, win=input_win)
    test_set   = set_wid(dataset=test_set, vocab=vocab, win=input_win)

    # train_set  = set_cid(dataset=train_set, char_vocab=char_vocab)
    # val_set    = set_cid(dataset=val_set, char_vocab=char_vocab)
    # tar_un_set = set_cid(dataset=tar_un_set, char_vocab=char_vocab)
    # test_set   = set_cid(dataset=test_set, char_vocab=char_vocab)

    train_set, ote_tag_vocab, ts_tag_vocab = set_labels(dataset=train_set, tagging_schema=tagging_schema)
    val_set, _, _ = set_labels(dataset=val_set, tagging_schema=tagging_schema)
    tar_un_set,  _, _ = set_labels(dataset=tar_un_set, tagging_schema=tagging_schema)
    test_set, _, _ = set_labels(dataset=test_set, tagging_schema=tagging_schema)

    train_set = set_lm_labels(dataset=train_set, vocab=vocab, stm_lex=stm_lex, stm_win=stm_win)
    val_set = set_lm_labels(dataset=val_set, vocab=vocab, stm_lex=stm_lex, stm_win=stm_win)
    tar_un_set = set_lm_labels(dataset=tar_un_set, vocab=vocab, stm_lex=stm_lex, stm_win=stm_win)
    test_set = set_lm_labels(dataset=test_set, vocab=vocab, stm_lex=stm_lex, stm_win=stm_win)

    # n_train = len(train_set)
    # # use 10% training data for dev experiment
    # n_val = int(n_train * 0.1)
    # # generate a uniform random sample from np.range(n_train) of size n_val
    # # This is equivalent to np.random.permutation(np.arange(n_train))[:n_val]
    #
    # val_sample_ids = np.random.choice(n_train, n_val, replace=False)
    # print("The first 15 validation samples:", val_sample_ids[:15])
    # val_set, tmp_train_set = [], []
    # for i in range(n_train):
    #     record = train_set[i]
    #     if i in val_sample_ids:
    #         val_set.append(record)
    #     else:
    #         tmp_train_set.append(record)
    # train_set = [r for r in tmp_train_set]

    return train_set, val_set, tar_un_set, test_set, vocab, char_vocab, ote_tag_vocab, ts_tag_vocab


def load_embeddings(path, vocab, source_domain, target_domain, emb_name):
    """
    load pre-trained word embeddings from the disk
    :param path: absolute path of the embedding files
    :param vocab: vocabulary
    :param ds_name: name of dataset
    :param emb_name: name of word embedding
    :return:
    """

    pkl = './work/embeddings/%s_%s_%s.pkl' % (source_domain, target_domain, emb_name)
    if os.path.exists(pkl):
        print("Load embeddings from existing pkl file %s..." % pkl)
        # word embeddings weights have been loaded
        embeddings = pickle.load(open(pkl, 'rb'))
    else:
        print("Load embedding from %s..." % path)
        raw_embeddings = {}
        if emb_name == 'yelp_electronics':
            with open(path) as fp:
                for line in fp:
                    word_vector = line.split(",")[:-1]
                    vector_list = []
                    for element in word_vector[len(word_vector) - 100:]:
                        vector_list.append(float(element))
                    word = ','.join(word_vector[:len(word_vector) - 100])
                    vector = np.asarray(vector_list)
                    if word in vocab:
                        raw_embeddings[word] = vector
        else:
            with open(path) as fp:
                for line in fp:
                    eles = line.strip().split(' ')
                    word = eles[0]
                    if word in vocab:
                        raw_embeddings[word] = eles[1:]

        dim_w = len(raw_embeddings['the'])
        n_words = len(vocab)
        embeddings = np.zeros(shape=(n_words, dim_w))
        for w in vocab:
            wid = vocab[w]
            if w in raw_embeddings:
                embeddings[wid] = np.array([float(ele) for ele in raw_embeddings[w]])
            else:
                # for OOV words, add random initialization
                embeddings[wid] = np.random.uniform(-0.25, 0.25, dim_w)
        print("Find %s word embeddings..." % len(embeddings))
        if not os.path.exists('./work/embeddings'):
            os.mkdir('./work/embeddings')
        emb_path = './work/embeddings/%s_%s_%s.pkl' % (source_domain, target_domain, emb_name)
        # write the embedding weights back to the disk
        pickle.dump(embeddings, open(emb_path, 'wb'))
    embeddings = np.array(embeddings, dtype='float32')
    return embeddings


def load_char_embeddings(char_vocab, ds_name):
    """
    load pre-trained character-level embeddings
    :param char_vocab: vocabulary of character
    :param ds_name: name of dataset
    :return:
    """
    n_char = len(char_vocab)
    pkl = './embeddings/%s_char.pkl' % ds_name
    if os.path.exists(pkl):
        print("Load character embeddings from %s..." % pkl)
        embeddings = pickle.load(open(pkl, 'rb'))
    else:
        emb_path = './embeddings/char-embeddings.txt'
        print("Load character embeddings from %s..." % emb_path)
        raw_embeddings = {}
        n_found = 0
        with open(emb_path) as fp:
            for line in fp:
                eles = line.strip().split()
                ch = eles[0]
                vec = [float(ele) for ele in eles[1:]]
                if ch not in raw_embeddings:
                    raw_embeddings[ch] = vec

        dim_ch = len(raw_embeddings['A'])
        embeddings = np.zeros(shape=(n_char, dim_ch))
        for ch in char_vocab:
            cid = char_vocab[ch]
            if ch in raw_embeddings:
                embeddings[cid] = np.array(raw_embeddings[ch])
                n_found += 1
            else:
                embeddings[cid] = np.random.uniform(-0.25, 0.25, dim_ch)
        print("Find %s chars in pre-trained character embeddings..." % n_found)
        embeddings = np.array(embeddings, dtype='float32')
        pickle.dump(embeddings, open(pkl, 'wb'))
    return embeddings


def label2tag(label_sequence, tag_vocab):
    """
    convert label sequence to tag sequence
    :param label_sequence: label sequence
    :param tag_vocab: tag vocabulary, i.e., mapping between tag and label
    :return:
    """
    inv_tag_vocab = {}
    for tag in tag_vocab:
        label = tag_vocab[tag]
        inv_tag_vocab[label] = tag
    tag_sequences = []
    n_tag = len(tag_vocab)
    for seq in label_sequence:
        tag_seq = []
        for l in seq:
            if l in inv_tag_vocab:
                tag_seq.append(inv_tag_vocab[l])
            elif l == n_tag or l == n_tag + 1:
                tag_seq.append("O")
            else:
                raise Exception("Invalid label %s" % l)
        tag_sequences.append(tag_seq)

    return tag_sequences


def tag2predictions(ote_tag_sequence, ts_tag_sequence):
    """
    transform BIEOS tag sequence to the list of aspects together with sentiment
    :param ote_tag_sequence: tag sequence for opinion target extraction
    :param ts_tag_sequence: tag sequence for targeted sentiment
    :return: a list of aspects/entities
    """
    n_tag = len(ote_tag_sequence)
    # opinion target sequence and targeted sentiment sequence
    ot_sequence, ts_sequence = [], []
    beg, end = -1, -1
    for i in range(n_tag):
        tag = ote_tag_sequence[i]
        if tag == 'S':
            ot_sequence.append((i, i))
        elif tag == 'B':
            beg = i
        elif tag == 'E':
            end = i
            if end > beg and beg != -1:
                ot_sequence.append((beg, end))
                beg, end = -1, -1
    sentiments = []
    beg, end = -1, -1
    for i in range(n_tag):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiments[0]))
            sentiments = []
        elif pos == 'B':
            beg = i
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1

            # schema2: only consider the sentiment at the beginning of the aspect span
            # if end > beg > -1:
            #    ts_sequence.append((beg, end, sentiments[0]))
            #    sentiments = []
            #    beg, end = -1, -1
    return ot_sequence, ts_sequence

def to_conll(train, val, test, embeddings, vocab, ds_name):
    """

    :param train: training dataset
    :param val: validation / development dataset
    :param test: testing dataset
    :param embeddings: pre-trained word embeddings
    :param vocab: vocabulary
    :return:
    """
    inv_vocab = {}
    for w in vocab:
        wid = vocab[w]
        inv_vocab[wid] = w
    train_lines = semeval2conll(dataset=train)
    dev_lines = semeval2conll(dataset=val)
    test_lines = semeval2conll(dataset=test)
    base_folder = '/projdata9/info_fil/lixin/Research/NCRFpp/sample_data'
    with open('%s/%s_train.txt' % (base_folder, ds_name), 'w+') as fp:
        fp.writelines(train_lines)
    with open('%s/%s_dev.txt' % (base_folder, ds_name), 'w+') as fp:
        fp.writelines(dev_lines)
    with open('%s/%s_test.txt' % (base_folder, ds_name), 'w+') as fp:
        fp.writelines(test_lines)

    emb_lines = []
    for i in range(len(embeddings)):
        word = inv_vocab[i]
        emb_vec = embeddings[i]
        emb_lines.append('%s %s\n' % (word, ' '.join([str(ele) for ele in emb_vec])))
    # write the embeddings back to the NCRFpp foler
    with open('%s/%s_emb.txt' % (base_folder, ds_name), 'w+') as fp:
        fp.writelines(emb_lines)


def semeval2conll(dataset):
    """
    transform the format of semeval datasets to conll form
    :param dataset: input dataset
    :return:
    """
    conll_lines = []
    for record in dataset:
        ote_raw_tags = record['ote_raw_tags']
        ts_raw_tags = record['ts_raw_tags']
        words = record['words']
        ote_tags, ts_tags = ot2bieos(ote_tag_sequence=ote_raw_tags, ts_tag_sequence=ts_raw_tags)
        for (w, t) in zip(words, ts_tags):
            conll_lines.append('%s %s\n' % (w, t))
        # use empty line to seprate the samples
        conll_lines.append('\n')
    return conll_lines

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]

def batch_generator(dataset, batch_size, shuffle=True):

    attrs = ['lm_labels', 'wids', 'length', 'ote_labels', 'ts_labels', 'opn_labels', 'stm_lm_labels', 'lm_labels_f', 'lm_labels_b']
    data = [dataset[a] for a in attrs]

    if shuffle:
        data = shuffle_aligned_list(data)
    num = data[0].shape[0]

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= num:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def transform_data_format(data):

    words = []
    ote_tags = []
    ts_tags = []
    lm_labels = []
    wids = []
    length = []
    ote_labels = []
    ts_labels = []
    opn_labels = []
    stm_lm_labels = []
    lm_labels_f = []
    lm_labels_b = []
    n_records = len(data)
    for i in range(n_records):
        words.append(data[i]['words'])
        ote_tags.append(data[i]['ote_tags'])
        ts_tags.append(data[i]['ts_tags'])
        lm_labels.append(data[i]['lm_labels'])
        wids.append(data[i]['wids'])
        length.append(data[i]['length'])
        ote_labels.append(data[i]['ote_labels'])
        ts_labels.append(data[i]['ts_labels'])
        opn_labels.append(data[i]['opn_labels'])
        stm_lm_labels.append(data[i]['stm_lm_labels'])
        lm_labels_f.append(data[i]['lm_labels_f'])
        lm_labels_b.append(data[i]['lm_labels_b'])

    lm_labels = np.array(lm_labels)
    wids = np.array(wids)
    length = np.array(length)
    ote_labels = np.array(ote_labels)
    ts_labels = np.array(ts_labels)
    opn_labels = np.array(opn_labels)
    stm_lm_labels = np.array(stm_lm_labels)
    lm_labels_f = np.array(lm_labels_f)
    lm_labels_b = np.array(lm_labels_b)

    data_trans = {}
    data_trans['words'] = words
    data_trans['ote_tags'] = ote_tags
    data_trans['ts_tags'] = ts_tags
    data_trans['lm_labels'] = lm_labels
    data_trans['wids'] = wids
    data_trans['length'] = length
    data_trans['ote_labels'] = ote_labels
    data_trans['ts_labels'] = ts_labels
    data_trans['opn_labels'] = opn_labels
    data_trans['stm_lm_labels'] = stm_lm_labels
    data_trans['lm_labels_f'] = lm_labels_f
    data_trans['lm_labels_b'] = lm_labels_b

    return data_trans

def get_train_feed_dict(model, reviews, win_reviews, batch_length, ote_labels, ts_labels, opn_labels, stm_lm_labels, lr, dropout_rate=1.0, train_flag=True):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update({model.reviews: reviews})
    feed_dict.update({model.win_reviews: win_reviews})
    feed_dict.update({model.batch_length: batch_length})
    feed_dict.update({model.asp_labels: ote_labels})
    feed_dict.update({model.ts_labels: ts_labels})
    feed_dict.update({model.opn_labels: opn_labels})
    feed_dict.update({model.stm_lm_labels: stm_lm_labels})
    # feed_dict.update({model.lmf_labels: lmf_labels})
    # feed_dict.update({model.lmb_labels: lmb_labels})
    feed_dict.update({model.lr: lr})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.train_flag: train_flag})

    return feed_dict

def get_eval_feed_dict(model, reviews, win_reviews, batch_length, opn_labels, dropout_rate=1.0, train_flag=False, domain_flag=True):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update({model.reviews: reviews})
    feed_dict.update({model.win_reviews: win_reviews})
    feed_dict.update({model.batch_length: batch_length})
    feed_dict.update({model.opn_labels: opn_labels})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.train_flag: train_flag})
    feed_dict.update({model.domain_flag: domain_flag})

    return feed_dict

def eval_metric(sess, model, params, dataset, domain_flag=True, eval_batch_size=100):

    gold_ote = dataset['ote_tags']
    gold_ts  = dataset['ts_tags']
    num = len(dataset['ote_tags'])

    ote_predictions = []
    ts_predictions  = []
    batch_count = 0
    batch_number = math.ceil(num / float(eval_batch_size))
    while True:
        if batch_count > batch_number:
            break
        start = batch_count * eval_batch_size
        end = start + eval_batch_size
        batch_count += 1

        feed_dict = get_eval_feed_dict(model,
                                       reviews=dataset['lm_labels'][start:end],
                                       win_reviews=dataset['wids'][start:end],
                                       batch_length=dataset['length'][start:end],
                                       opn_labels=dataset['opn_labels'][start:end],
                                       dropout_rate=1.0,
                                       train_flag=False,
                                       domain_flag=domain_flag)

        ote_prediction, ts_prediction = sess.run([model.asp_predictions, model.ts_predictions], feed_dict=feed_dict)
        ote_predictions.extend(ote_prediction.tolist())
        ts_predictions.extend(ts_prediction.tolist())


    ote_predictions = label2tag(label_sequence=ote_predictions, tag_vocab=params.ote_tag_vocab)
    ts_predictions  = label2tag(label_sequence=ts_predictions, tag_vocab=params.ts_tag_vocab)

    # evaluation
    ote_scores = evaluate_ote(gold_ot=gold_ote, pred_ot=ote_predictions, length=dataset['length'])
    ts_scores = evaluate_ts(gold_ts=gold_ts, pred_ts=ts_predictions, length=dataset['length'])


    return ote_scores, ts_scores, ote_predictions, ts_predictions
    # return ts_scores, ts_predictions


def store_results(model_name, round, source_domain, target_domain, ote_score, ts_score):

    output_dir = "./work/results/"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_dir = output_dir + "%s/" % model_name
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    ote_p, ote_r, ote_f1 = ote_score
    _, ts_micro_p, ts_micro_r, ts_micro_f1 = ts_score

    f = open(model_dir+'Round%d.txt' % (round), "a")
    f.write('{:<20} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(source_domain+'-'+target_domain,
                                                                        ote_p, ote_r, ote_f1,
                                                                        ts_micro_p, ts_micro_r, ts_micro_f1))
    f.close()

def store_predictions(params, test_set, best_pred_ote, best_pred_ts, test_ts_micro_f1):

    n_sample = len(test_set['lm_labels'])
    gold_ote = test_set['ote_tags']
    gold_ts = test_set['ts_tags']
    output_lines = ['Source domain: %s  ' % params.source_domain, 'Target domain: %s\n' % params.target_domain,
                    'Parameter settings: \n']
    params_dict = vars(params)
    for k in params_dict:
        if k == 'char_vocab' or k == 'vocab':
            continue
        else:
            v = params_dict[k]
            output_lines.append('\t%s: %s\n' % (k, v))
    output_lines.append("==============================================\n\n")
    for i in range(n_sample):
        ote_seq = best_pred_ote[i]
        ts_seq = best_pred_ts[i]
        w_seq = test_set['words'][i]
        length = test_set['length'][i]
        ote_seq_gold = gold_ote[i]
        ts_seq_gold = gold_ts[i]
        assert len(ote_seq) == len(ts_seq) == len(w_seq)
        for j in range(length):
            word = w_seq[j]
            ote_tag = ote_seq[j]
            ote_tag_gold = ote_seq_gold[j]
            ts_tag = ts_seq[j]
            ts_tag_gold = ts_seq_gold[j]
            output_lines.append('%20s\t%s\t%s\t%s\t%s\n' % (word, ote_tag, ote_tag_gold, ts_tag, ts_tag_gold))
        # use empty lines as the separator
        output_lines.append('\n')

    output_dir = './work/predictions'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, params.model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    output_file = os.path.join(output_dir, '%s_%s_%.6lf.txt' % (
        params.source_domain, params.target_domain, test_ts_micro_f1))
    with open(output_file, 'w+') as fp:
        fp.writelines(output_lines)