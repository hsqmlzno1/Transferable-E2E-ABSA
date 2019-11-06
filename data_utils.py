import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
import os
from collections import defaultdict as dd
import random
import math


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_corpus(args, cfgs):


    source_domain = args.source_domain
    target_domain = args.target_domain

    names = ['x', 'y', 'tx', 'ty', 'allx_s', 'ally_s', 'allx_t', 'ally_t', 'adj_s',  'adj_t']
    objects = []
    for i in range(len(names)):
        with open("data/graph-data/%s_%s/ind.%s" %(source_domain, target_domain, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx_s, ally_s, allx_t, ally_t, adj_s, adj_t = tuple(objects)

    print(x.shape, y.shape, tx.shape, ty.shape, allx_s.shape, ally_s.shape, allx_t.shape, ally_t.shape, adj_s.shape, adj_t.shape)

    src_features = allx_s.tolil()
    src_labels   = ally_s

    tar_features = allx_t.tolil()
    tar_labels   = ally_t

    train_size = 1600
    val_size = 400
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val   = range(len(y), len(y) + val_size)
    idx_test  = range(len(ty))

    train_mask = sample_mask(idx_train, src_labels.shape[0])
    val_mask   = sample_mask(idx_val,   src_labels.shape[0])
    test_mask  = sample_mask(idx_test,  tar_labels.shape[0])

    y_train = np.zeros(src_labels.shape)
    y_val   = np.zeros(src_labels.shape)
    y_test  = np.zeros(tar_labels.shape)
    y_train[train_mask, :] = src_labels[train_mask, :]
    y_val[val_mask, :]     = src_labels[val_mask, :]
    y_test[test_mask, :]   = tar_labels[test_mask, :]

    adj_s = adj_s + adj_s.T.multiply(adj_s.T > adj_s) - adj_s.multiply(adj_s.T > adj_s)
    adj_t = adj_t + adj_t.T.multiply(adj_t.T > adj_t) - adj_t.multiply(adj_t.T > adj_t)

    return adj_s, adj_t, src_features, tar_features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def load_corpus2(args, cfgs):

    source_domain = args.source_domain
    target_domain = args.target_domain
    data_dir = 'data/graph-data-%s_%s/' % (args.embed_type, str(args.keep_size))

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'ally_aux_p', 'ally_aux_n', 'adj', 'adj_mask', 'graph', 'bow', 'pos_pivots_ind', 'neg_pivots_ind']
    objects = []
    for i in range(len(names)):
        with open(data_dir+"%s_%s/ind.%s" %(source_domain, target_domain, names[i]), 'rb') as f:
        # with open("/qydata/zlict/transfer-learning/GCN/text_gcn/data/ind.%s_%s.%s" % (source_domain, target_domain, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, ally_aux_p, ally_aux_n, adj, adj_mask, graph, bow, pos_pivots_ind, neg_pivots_ind = tuple(objects)

    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape, ally_aux_p.shape, ally_aux_n.shape, adj.shape, adj_mask.shape, bow.shape)

    # features = allx.tolil()
    features = np.matrix(allx.todense(), dtype=np.float32)
    labels   = ally
    aux_p_labels   = ally_aux_p
    aux_n_labels   = ally_aux_n
    print(features)

    # features = sp.vstack((allx, tx)).tolil()
    # labels = np.vstack((ally, ty))

    train_size = 1600
    val_size = 400
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val   = range(len(y), len(y) + val_size)
    idx_test  = range(len(y) + val_size, len(y) + val_size + test_size)
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y) + val_size)
    # idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask   = sample_mask(idx_val,   labels.shape[0])
    test_mask  = sample_mask(idx_test,  labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val   = np.zeros(labels.shape)
    y_test  = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :]     = labels[val_mask, :]
    y_test[test_mask, :]   = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_mask = adj_mask + adj_mask.T.multiply(adj_mask.T > adj_mask) - adj_mask.multiply(adj_mask.T > adj_mask)

    vocab = load_vocab(data_dir, source_domain, target_domain)

    pos_pivots_ind = [12000 + ind for ind in pos_pivots_ind]
    neg_pivots_ind = [12000 + ind for ind in neg_pivots_ind]

    return x.todense(), y, adj, adj_mask, features, graph, bow.todense(), labels, aux_p_labels, aux_n_labels, \
           pos_pivots_ind, neg_pivots_ind, \
           y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size, vocab

def load_corpus3(args, cfgs):

    source_domain = args.source_domain
    target_domain = args.target_domain
    data_dir = 'data/graph-data-%s_%s/' % (args.embed_type, str(args.keep_size))

    names = ['x', 'y', 'allx', 'ally', 'ally_aux_p', 'ally_aux_n', 'adj', 'adj_mask', 'graph', 'bow', 'bow_mask', 'pos_pivots_ind', 'neg_pivots_ind', 'src_spec_ind', 'tar_spec_ind']
    objects = []
    for i in range(len(names)):
        with open(data_dir+"%s_%s/ind.%s" %(source_domain, target_domain, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, allx, ally, ally_aux_p, ally_aux_n, adj, adj_mask, graph, bow, bow_mask, pos_pivots_ind, neg_pivots_ind, src_spec_ind, tar_spec_ind = tuple(objects)

    print(allx.shape, ally.shape, ally_aux_p.shape, ally_aux_n.shape, adj.shape, adj_mask.shape, bow.shape, bow_mask.shape)

    n2v = np.matrix(allx.todense(), dtype=np.float32)
    labels   = ally
    aux_p_labels   = ally_aux_p
    aux_n_labels   = ally_aux_n

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_mask = adj_mask + adj_mask.T.multiply(adj_mask.T > adj_mask) - adj_mask.multiply(adj_mask.T > adj_mask)

    vocab = load_vocab(data_dir, source_domain, target_domain)

    src_spec_ind = [12000 + ind for ind in src_spec_ind]
    tar_spec_ind = [12000 + ind for ind in tar_spec_ind]
    pos_pivots_ind = [12000 + ind for ind in pos_pivots_ind]
    neg_pivots_ind = [12000 + ind for ind in neg_pivots_ind]

    return x.todense(), y, n2v, adj, adj_mask, graph, bow.todense(), bow_mask, labels, aux_p_labels, aux_n_labels, pos_pivots_ind, neg_pivots_ind, src_spec_ind, tar_spec_ind, vocab

def load_corpus_couple(args, cfgs):

    source_domain = args.source_domain
    target_domain = args.target_domain
    data_dir = 'data/graph-data-couple-%s_%s/' % (args.embed_type, str(args.keep_size))

    names = ['x', 'y', 'allx', 'ally', 'ally_aux_p', 'ally_aux_n',
             'src_adj', 'src_adj_mask', 'src_graph', 'src_bow',
             'tar_adj', 'tar_adj_mask', 'tar_graph', 'tar_bow',
             'pos_pivots_ind', 'neg_pivots_ind']
    objects = []
    for i in range(len(names)):
        with open(data_dir+"%s_%s/ind.%s" %(source_domain, target_domain, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, allx, ally, ally_aux_p, ally_aux_n, \
    src_adj, src_adj_mask, src_graph, src_bow, \
    tar_adj, tar_adj_mask, tar_graph, tar_bow, \
    pos_pivots_ind, neg_pivots_ind = tuple(objects)

    print(allx.shape, ally.shape, ally_aux_p.shape, ally_aux_n.shape,
          src_adj.shape, src_adj_mask.shape, src_bow.shape,
          tar_adj.shape, tar_adj_mask.shape, tar_bow.shape)

    n2v = np.matrix(allx.todense(), dtype=np.float32)
    labels   = ally
    aux_p_labels   = ally_aux_p
    aux_n_labels   = ally_aux_n

    src_adj = src_adj + src_adj.T.multiply(src_adj.T > src_adj) - src_adj.multiply(src_adj.T > src_adj)
    src_adj_mask = src_adj_mask + src_adj_mask.T.multiply(src_adj_mask.T > src_adj_mask) - src_adj_mask.multiply(src_adj_mask.T > src_adj_mask)
    tar_adj = tar_adj + tar_adj.T.multiply(tar_adj.T > tar_adj) - tar_adj.multiply(tar_adj.T > tar_adj)
    tar_adj_mask = tar_adj_mask + tar_adj_mask.T.multiply(tar_adj_mask.T > tar_adj_mask) - tar_adj_mask.multiply(tar_adj_mask.T > tar_adj_mask)

    vocab = load_vocab(data_dir, source_domain, target_domain)

    pos_pivots_ind = [6000 + ind for ind in pos_pivots_ind]
    neg_pivots_ind = [6000 + ind for ind in neg_pivots_ind]

    return x.todense(), y, n2v, \
           src_adj, src_adj_mask, src_graph, src_bow.todense(), \
           tar_adj, tar_adj_mask, tar_graph, tar_bow.todense(), \
           labels, aux_p_labels, aux_n_labels, pos_pivots_ind, neg_pivots_ind, vocab

def load_vocab(output_dir, source_domain, target_domain):

    vocab = {}
    with open(output_dir+"%s_%s/vocab.txt" % (source_domain, target_domain), 'rb') as f:
        for line in f.readlines():
            word, idx = line.strip().split()
            vocab[word] = idx
    return vocab


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def preprocess_features2(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.todense()
    # return adj_normalized


def construct_feed_dict(model, features_s, features_t, support_s, support_t, labels, labels_mask, dropout_rate, mode):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({model.features_s: features_s})
    feed_dict.update({model.features_t: features_t})
    feed_dict.update({model.support_s: support_s})
    feed_dict.update({model.support_t: support_t})
    feed_dict.update({model.labels: labels})
    feed_dict.update({model.labels_mask: labels_mask})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.mode: mode})

    return feed_dict

def construct_feed_dict2(model, features, support, labels, labels_mask, dropout_rate, mode):
    """Construct feed dictionary."""
    feed_dict = dict()
    # feed_dict.update({model.features: features})
    feed_dict.update({model.support: support})
    feed_dict.update({model.labels: labels})
    feed_dict.update({model.labels_mask: labels_mask})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.mode: mode})

    return feed_dict

def construct_feed_dict3(model, support, support_mask, ind, labels, p_labels, n_labels, dropout_rate, mode):
    """Construct feed dictionary."""
    # features = sp.identity(support[2][1])
    # features = preprocess_features(features)
    feed_dict = dict()
    # feed_dict.update({model.features: features})
    # feed_dict.update({model.support: support})
    # feed_dict.update({model.support_mask: support_mask})
    # feed_dict.update({model.ind: ind})
    feed_dict.update({model.labels: labels})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.mode: mode})
    if p_labels is not None and n_labels is not None:
        feed_dict.update({model.p_labels: p_labels})
        feed_dict.update({model.n_labels: n_labels})

    return feed_dict

def construct_feed_dict4(model, x, x_mask, y, support, support_mask, ind, p_ind, py, p_labels, n_labels, tar_spec_ind, dropout_rate=1.0, train_flag=False, domain_flag=True):
    """Construct feed dictionary."""

    features = sp.identity(support.shape[0])
    features = preprocess_features(features)

    feed_dict = dict()
    feed_dict.update({model.x: x})
    feed_dict.update({model.x_mask: x_mask})
    feed_dict.update({model.y: y})
    feed_dict.update({model.features: features})
    # feed_dict.update({model.support: support})
    # feed_dict.update({model.support_mask: support_mask})
    feed_dict.update({model.ind: ind})
    feed_dict.update({model.wt_ind: tar_spec_ind})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.train_flag: train_flag})
    feed_dict.update({model.domain_flag: domain_flag})

    if p_labels is not None and n_labels is not None:
        feed_dict.update({model.p_labels: p_labels})
        feed_dict.update({model.n_labels: n_labels})

    if p_ind is not None and py is not None:
        feed_dict.update({model.p_ind: p_ind})
        feed_dict.update({model.py: py})

    return feed_dict

def construct_feed_dict5(model, x, y, ind, p_labels, n_labels, dropout_rate, train_flag, domain_flag):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update({model.x: x})
    feed_dict.update({model.y: y})
    feed_dict.update({model.ind: ind})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.train_flag: train_flag})
    feed_dict.update({model.domain_flag: domain_flag})

    if p_labels is not None and n_labels is not None:
        feed_dict.update({model.p_labels: p_labels})
        feed_dict.update({model.n_labels: n_labels})

    return feed_dict

def construct_feed_dict6(model, x, y, features, ind, p_labels, n_labels, dropout_rate, train_flag):
    """Construct feed dictionary."""

    feed_dict = dict()
    feed_dict.update({model.x: x})
    feed_dict.update({model.y: y})
    feed_dict.update({model.features: features})
    feed_dict.update({model.ind: ind})
    feed_dict.update({model.dropout_rate: dropout_rate})
    feed_dict.update({model.train_flag: train_flag})

    if p_labels is not None and n_labels is not None:
        feed_dict.update({model.p_labels: p_labels})
        feed_dict.update({model.n_labels: n_labels})

    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_w2vec(vocab, args):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    if args.embed_type == "200dw":
        word_vecs = load_glove_vec('/qydata/zlict/tools/glove/wikipedia_6B/glove.6B.200d.txt', vocab)
    elif args.embed_type == "200dt":
        word_vecs = load_glove_vec('/qydata/zlict/tools/glove/twitter_27B/glove.twitter.27B.200d.txt', vocab)
    elif args.embed_type == "300dg":
        word_vecs = load_glove_vec('/qydata/zlict/tools/glove/commonCrawl_840B/glove.840B.300d.txt', vocab)
    elif args.embed_type == "300dw":
        word_vecs = load_bin_vec("/qydata/zlict/deep-learning/CNN_sentence/GoogleNews-vectors-negative300.bin", vocab)

    add_unknown_words(word_vecs, vocab)
    dim = word_vecs.values()[0].shape[0]
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    idx_word_map = dict()

    W = np.zeros(shape=(vocab_size, dim), dtype='float32')
    for word, idx in vocab.items():
        W[idx] = word_vecs[str(word)]
        word_idx_map[word] = idx
        idx_word_map[idx] = word

    return W, word_idx_map, idx_word_map

def load_glove_vec(fname, vocab):

    word_vecs = {}
    with open(fname, 'r') as f:
        for line in f:
            content = line.strip().split()
            word = content[0]
            if word in vocab:
                word_vecs[word] = np.array(map(float, content[1:]))
    return word_vecs

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    dim = word_vecs.values()[0].shape[0]
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, dim)


def gen_graph(graph, cfgs):
    """generator for batches for graph context loss.
    """

    num_ver = max(graph.keys()) + 1
    # src_inst_ind = range(2000) + range(4000, 8000)
    # tar_inst_ind = range(2000, 4000) + range(8000, 12000)

    while True:
        ind = np.random.permutation(num_ver)
        i = 0
        while i < ind.shape[0]:
            g, gy = [], []
            j = min(ind.shape[0], i + cfgs.g_batch_size)
            for k in ind[i: j]:
                if len(graph[k]) == 0: continue
                path = [k]
                for _ in range(cfgs.path_size):
                    path.append(random.choice(graph[path[-1]]))
                for l in range(len(path)):
                    for m in range(l - cfgs.window_size, l + cfgs.window_size + 1):
                        if m < 0 or m >= len(path): continue
                        # if (path[l] in src_inst_ind and path[m] in tar_inst_ind) or (path[m] in src_inst_ind and path[l] in tar_inst_ind):
                        g.append([path[l], path[m]])
                        gy.append(1.0)
                        for _ in range(cfgs.neg_samp):
                            g.append([path[l], random.randint(0, num_ver - 1)])
                            gy.append(- 1.0)
            yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
            i = j

def gen_label_graph(x, y, cfgs):
    """generator for batches for label context loss.
    """
    print(x.shape)
    print(y.shape)
    labels, label2inst, not_label = [], dd(list), dd(list)
    for i in range(x.shape[0]):
        flag = False
        for j in range(y.shape[1]):
            if y[i, j] == 1 and not flag:
                labels.append(j)
                label2inst[j].append(i)
                flag = True
            elif y[i, j] == 0:
                not_label[j].append(i)

    while True:
        g, gy = [], []
        for _ in range(cfgs.g_sample_size):
            x1 = random.randint(0, x.shape[0] - 1)
            label = labels[x1]
            if len(label2inst) == 1: continue
            x2 = random.choice(label2inst[label])
            g.append([x1, x2])
            gy.append(1.0)
            for _ in range(cfgs.neg_samp):
                g.append([x1, random.choice(not_label[label])])
                gy.append(- 1.0)
        yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)

# def gen_graph(graph, y, cfgs):
#     """generator for batches for graph context loss.
#     """
#
#     label_size = 1600
#     num_ver = max(graph.keys()) + 1
#
#     while True:
#         ind = np.random.permutation(label_size)
#         i = 0
#         while i < ind.shape[0]:
#             g, gy = [], []
#             j = min(ind.shape[0], i + cfgs.g_batch_size)
#             for k in ind[i: j]:
#                 if len(graph[k]) == 0: continue
#                 path = [k]
#                 for _ in range(cfgs.path_size):
#                     path.append(random.choice(graph[path[-1]]))
#                 for l in range(1, len(path)):
#                     g.append([path[0], path[l]])
#                     gy.append(y[path[0]])
#             yield np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)
#             i = j

# def gen_graph(graph, ind, y, cfgs):
#     """generator for batches for graph context loss.
#     """
#     gs, gt, gy = [], [], []
#     num_ver = max(graph.keys()) + 1
#     eos = num_ver
#     for k in ind:
#         if len(graph[k]) == 0: continue
#         path = [k]
#         for _ in range(cfgs.path_size):
#             path.append(random.choice(graph[path[-1]]))
#         gs.append(path)
#         gt.append(path+[eos])
#         gy.append(y[path[0]])
#
#     return np.array(gs, dtype=np.int32), np.array(gt, dtype=np.int32), np.array(gy, dtype=np.float32)

# def gen_graph2(graph, ind, cfgs):
#     """generator for batches for graph context loss.
#     """
#     gs, gt = [], []
#     num_ver = max(graph.keys()) + 1
#     eos = num_ver
#     for k in ind:
#         if len(graph[k]) == 0: continue
#         path = [k]
#         for _ in range(cfgs.path_size):
#             path.append(random.choice(graph[path[-1]]))
#         gs.append(path)
#         gt.append(path+[eos])
#
#     return np.array(gs, dtype=np.int32), np.array(gt, dtype=np.int32)

# def gen_tar_graph(graph, ind, pos_pivots_ind, neg_pivots_ind, cfgs):
#     """generator for batches for graph context loss.
#     """
#
#     # print('Node %d:' % path[-1])
#     # print('Pos neighbor num: %d' % len(pos_neighbor), pos_neighbor)
#     # print('Neg neighbor num: %d' % len(neg_neighbor), neg_neighbor)
#
#     g, gy = [], []
#     tar_range = set(range(2000, 4000) + range(8000, 12000))
#     # num_ver = max(graph.keys()) + 1
#     # eos = num_ver
#     for i in ind:
#         neighbor = graph[i]
#         if len(neighbor) == 0:
#             continue
#         pos_neighbor = set(neighbor) & set(pos_pivots_ind)
#         neg_neighbor = set(neighbor) & set(neg_pivots_ind)
#         if len(pos_neighbor) > len(neg_neighbor) and len(pos_neighbor) > 0:
#             pos_node = []
#             for j in pos_neighbor:
#                 pos_node += graph[j]
#             pos_node = set(pos_node) & tar_range
#             neg_node = tar_range - pos_node
#             g.append([i, random.choice(list(pos_node))])
#             g.append([i, random.choice(list(neg_node))])
#             # gy.append(1.0)
#             # gy.append(-1.0)
#             gy.append([1,0])
#             gy.append([0,1])
#
#         if len(neg_neighbor) > len(pos_neighbor) and len(neg_neighbor) > 0:
#             pos_node = []
#             for j in neg_neighbor:
#                 pos_node += graph[j]
#             pos_node = set(pos_node) & tar_range
#             neg_node = tar_range - pos_node
#             g.append([i, random.choice(pos_node)])
#             g.append([i, random.choice(neg_node)])
#             gy.append([1,0])
#             gy.append([0,1])
#
#     return np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)

def gen_pivot_graph(graph, ind, pos_pivots_ind, neg_pivots_ind, cfgs):

    g = {}
    tar_range = set(range(2000, 4000) + range(8000, 12000))
    for i in ind:
        neighbor = graph[i]
        if len(neighbor) == 0:
            continue
        pos_neighbor = set(neighbor) & set(pos_pivots_ind)
        neg_neighbor = set(neighbor) & set(neg_pivots_ind)
        if len(pos_neighbor) > len(neg_neighbor) and len(pos_neighbor) > 0:
            pos_node = []
            for j in pos_neighbor:
                pos_node += graph[j]
            pos_node = set(pos_node) & tar_range
            neg_node = tar_range - pos_node
            g[i] = [list(pos_node), list(neg_node)]
        if len(neg_neighbor) > len(pos_neighbor) and len(neg_neighbor) > 0:
            pos_node = []
            for j in neg_neighbor:
                pos_node += graph[j]
            pos_node = set(pos_node) & tar_range
            neg_node = tar_range - pos_node
            g[i] = [list(pos_node), list(neg_node)]

    return g

def gen_pivot_graph2(graph, pos_pivots_ind, neg_pivots_ind, a_ind_range, b_ind_range):

    g = {}
    for i in pos_pivots_ind+neg_pivots_ind:
        neighbor = graph[i]
        if len(neighbor) == 0:
            continue
        pos_neighbor = set(neighbor) & set(a_ind_range)
        neg_neighbor = set(neighbor) & set(b_ind_range)
        if len(pos_neighbor) > 0 and len(neg_neighbor) > 0:
            g[i] = [list(pos_neighbor), list(neg_neighbor)]

    return g

def gen_random_walk(graph, ind, pos_pivots_ind, neg_pivots_ind ):

    g, gy, labels = [], [], []

    for i in ind:
        if i not in graph:
            continue
        pos_neighbor, neg_neighbor = graph[i][0], graph[i][1]
        # for j in range(3):
        g.append([random.choice(pos_neighbor), random.choice(neg_neighbor)])
        gy.append(i)
        if i in pos_pivots_ind:
            labels.append([1, 0])
        if i in neg_pivots_ind:
            labels.append([0, 1])

    return np.array(g, dtype=np.int32), np.array(gy, dtype=np.int32), np.array(labels, dtype=np.int32)

def gen_tar_graph(graph, ind):

    g, gy = [], []

    for i in ind:
        if i not in graph:
            continue
        pos_neighbor, neg_neighbor = graph[i][0], graph[i][1]
        g.append([i, random.choice(pos_neighbor)])
        g.append([i, random.choice(neg_neighbor)])
        gy.append(1.0)
        gy.append(-1.0)
        # gy.append([1,0])
        # gy.append([0,1])

    return np.array(g, dtype=np.int32), np.array(gy, dtype=np.float32)

def csr_2_sparse_tensor_tuple(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    indices = np.transpose(np.vstack((coo_matrix.row, coo_matrix.col)))
    values = coo_matrix.data
    shape = csr_matrix.shape
    return indices, values, shape

def shuffle_aligned_list(data):
    # num = data[0].shape[0]
    num = min([d.shape[0] for d in data])
    shuffle_index = np.random.permutation(num)
    return [d[shuffle_index] for d in data]

def batch_generator(data, batch_size, shuffle=True):
    if shuffle:
        data = shuffle_aligned_list(data)
    min_size = min([d.shape[0] for d in data])

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= min_size:
            batch_count = 0
            if shuffle:
                data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def sample_by_label(images, labels, num_labels, seed=None):
    """Extract equal number of sampels per class."""
    res = []
    rng = np.random.RandomState(seed=seed)
    for i in xrange(num_labels):
        a = images[labels == i]
        res.append(a)

    return res

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""
def adj_to_bias(adj, sizes, nhood=1):

    mt = np.eye(adj.shape[1])
    for _ in range(nhood):
        mt = np.matmul(mt, (adj + np.eye(adj.shape[1])))
    for i in range(sizes[0]):
        for j in range(sizes[0]):
            if mt[i,j] > 0.0:
                mt[i,j] = 1.0

    return -1e9 * (1.0 - mt)

# def adj_to_bias(adj, sizes, nhood=1):
#     nb_graphs = adj.shape[0]  # num_node
#     mt = np.empty(adj.shape)
#     for g in range(nb_graphs):
#         mt[g] = np.eye(adj.shape[1])
#         for _ in range(nhood):
#             mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
#         for i in range(sizes[g]):
#             for j in range(sizes[g]):
#                 if mt[g][i][j] > 0.0:
#                     mt[g][i][j] = 1.0
#
#     return -1e9 * (1.0 - mt)
