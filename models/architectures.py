
import tensorflow as tf
from nn_utils import *
import numpy as np
import nn_utils

class SuperNN(object):

    def __init__(self,
                 args,
                 word2vec,
                 init=None,
                 scope=None):

        self.args = args
        self.word2vec = word2vec
        self.init = init
        self.scope = scope

        self.hops = args.hops
        self.max_len = args.max_len
        self.input_win = args.input_win

        self.dim_w = args.dim_w
        self.dim_asp_h = args.dim_asp_h
        self.dim_opn_h = args.dim_opn_h
        self.dim_ts_h = args.dim_ts_h
        self.dim_rel = args.dim_rel
        self.dim_ote_y = args.dim_ote_y
        self.dim_ts_y = args.dim_ts_y
        self.dim_lm_y = args.dim_lm_y
        self.ote_tag_vocab = args.ote_tag_vocab
        self.ts_tag_vocab = args.ts_tag_vocab

        self.build_vars()

    def build_vars(self):

        with tf.variable_scope(self.scope):

            self.Wa = tf.Variable(self.init([2*self.dim_asp_h+2*self.dim_asp_h, 2*self.dim_asp_h]), name='Wa')
            self.Wo = tf.Variable(self.init([2*self.dim_asp_h+2*self.dim_opn_h, 2*self.dim_asp_h]), name='Wo')
            self.bias_a = tf.Variable(self.init([2*self.dim_asp_h, ]), name='bias_a')
            self.bias_o = tf.Variable(self.init([2*self.dim_asp_h, ]), name='bias_o')

            # relation matrices between aspect&aspect, opinion&opinion, aspect&opinion
            self.Ta  = tf.Variable(self.init([self.dim_rel, 2*self.dim_asp_h, 2*self.dim_asp_h]), name='Ta')
            self.To  = tf.Variable(self.init([self.dim_rel, 2*self.dim_opn_h, 2*self.dim_opn_h]), name='To')
            self.Tao = tf.Variable(self.init([self.dim_rel, 2*self.dim_asp_h, 2*self.dim_opn_h]), name='Tao')

            self.va = tf.Variable(self.init([2*self.dim_rel]), name='v_a')
            self.vo = tf.Variable(self.init([2*self.dim_rel]), name='v_o')


    def __call__(self, win_reviews, batch_length, ma_0, mo_0, dropout_rate, reuse=False):

        with tf.variable_scope(self.scope, reuse=reuse):

            with tf.variable_scope("Embedding_layer"):
                mask = tf.cast(tf.sign(win_reviews), tf.float32)
                self.input_emb = tf.nn.embedding_lookup(self.word2vec, win_reviews) #(b, m, win, d)
                self.input_emb = tf.nn.dropout(self.input_emb, dropout_rate)

            with tf.variable_scope('LSTM-OTE'):
                fw_cell = tf.contrib.rnn.LSTMCell(self.dim_asp_h)
                bw_cell = tf.contrib.rnn.LSTMCell(self.dim_asp_h)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, self.input_emb,
                                                                  sequence_length=batch_length,
                                                                  dtype=tf.float32)
                asp_h = tf.concat(outputs, -1) #(b, m, 2*dim_asp_h)

            with tf.variable_scope('Attention'):

                ma_t = tf.tile(ma_0, [tf.shape(asp_h)[0], 1]) #(b, 2*dim_asp_h)
                mo_t = tf.tile(mo_0, [tf.shape(asp_h)[0], 1]) #(b, 2*dim_opn_h)

                za_list, zo_list = [], []
                ma_list, mo_list = [], []

                ma_list.append(ma_t)  # (b, 2*dim_asp_h)
                mo_list.append(mo_t)  # (b, 2*dim_opn_h)

                for l in range(self.hops):

                    za = tf.concat([self.tensor_product(asp_h, ma_t, self.Ta,  self.Wa,  self.bias_a, dropout_rate),    #(b, m, 2*dim_rel)
                                    self.tensor_product(asp_h, mo_t, self.Tao, self.Wo,  self.bias_o, dropout_rate)],   #(b, m, 2*dim_rel)
                                    -1)  #(b, m, 2*dim_rel)

                    zo = tf.concat([self.tensor_product(asp_h, mo_t, self.To,  self.Wo, self.bias_o, dropout_rate),    #(b, m, 2*dim_rel)
                                    self.tensor_product(asp_h, ma_t, tf.transpose(self.Tao, [0, 2, 1]), self.Wa, self.bias_a, dropout_rate)],   #(b, m, 2*dim_rel)
                                    -1)  #(b, m, 2*dim_rel)

                    za_l = tf.reshape(za, [-1, 2*self.dim_rel]) #(b*m, 2*dim_rel)
                    zo_l = tf.reshape(zo, [-1, 2*self.dim_rel]) #(b*m, 2*dim_rel)

                    za_list.append(za_l)
                    zo_list.append(zo_l)

                    ea_l = tf.reduce_sum(tf.multiply(za_l, self.va), -1)    #(b*m)
                    eo_l = tf.reduce_sum(tf.multiply(zo_l, self.vo), -1)    #(b*m)

                    ea_l = tf.reshape(ea_l, [-1, self.max_len]) #(b,m)
                    eo_l = tf.reshape(eo_l, [-1, self.max_len]) #(b,m)

                    alpha_a = tf.expand_dims(nn_utils.mask_softmax(ea_l, axis=1, mask=mask), -1) #(b,m,1)
                    alpha_o = tf.expand_dims(nn_utils.mask_softmax(eo_l, axis=1, mask=mask), -1) #(b,m,1)

                    a_summary = tf.reduce_sum(asp_h * alpha_a, 1) #(b, 2*dim_ote_h)
                    ma_t = ma_t + a_summary

                    o_summary = tf.reduce_sum(asp_h * alpha_o, 1) #(b, 2*dim_ote_h)
                    mo_t = mo_t + o_summary

                    ma_list.append(ma_t) #(b, 2*dim_asp_h)
                    mo_list.append(mo_t) #(b, 2*dim_opn_h)

                asp_h = za_list[-1]
                opn_h = zo_list[-1]
                asp_h = tf.reshape(asp_h, [-1, self.max_len, 2*self.dim_rel])
                opn_h = tf.reshape(opn_h, [-1, self.max_len, 2*self.dim_rel])

            with tf.variable_scope('LSTM-TS'):
                fw_cell = tf.contrib.rnn.LSTMCell(self.dim_ts_h)
                bw_cell = tf.contrib.rnn.LSTMCell(self.dim_ts_h)
                outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, asp_h,
                                                                  sequence_length=batch_length,
                                                                  dtype=tf.float32)
                ts_h = tf.concat(outputs, -1) #(b, m, 2*dim_opn_h)

            with tf.variable_scope('FC_layer'):

                asp_h = tf.nn.dropout(asp_h, dropout_rate)   #(b, m,  2*dim_ote)
                opn_h = tf.nn.dropout(opn_h, dropout_rate)   #(b, m,  2*dim_ts)
                ts_h  = tf.nn.dropout(ts_h, dropout_rate)    #(b, m,  2*dim_asp_h+2*dim_opn_h)

                asp_h = tf.reshape(asp_h, [-1, asp_h.shape.as_list()[-1]])
                opn_h = tf.reshape(opn_h, [-1, opn_h.shape.as_list()[-1]])
                ts_h  = tf.reshape(ts_h,  [-1, ts_h.shape.as_list()[-1]])

                asp_pred = nn_utils.fc_layer(asp_h, output_dim=self.dim_ote_y, scope="asp_tagger", reuse=reuse)
                opn_pred = nn_utils.fc_layer(opn_h, output_dim=self.dim_lm_y,  scope="opn_tagger", reuse=reuse)
                ts_pred  = nn_utils.fc_layer(ts_h, output_dim=self.dim_ts_y, scope="ts_tagger", reuse=reuse)

        return asp_h, ts_h, asp_pred, opn_pred, ts_pred, a_summary, o_summary, tf.squeeze(alpha_a, 2), tf.squeeze(alpha_o, 2)

    def tensor_product(self, a, b, T, W, bias, dropout_rate, activation=None):

        _, element_size, a_dim = a.shape.as_list()
        _, b_dim = b.shape.as_list()

        emb_dim = T.shape.as_list()[-1]
        T_dropout = tf.nn.dropout(T, dropout_rate)

        a_re = tf.reshape(a, [-1, a_dim])  #(b*m, d_a)
        b_tile = tf.tile(tf.expand_dims(b, 1), [1, element_size, 1])           #(b, m, d_b)
        ab_fusion = tf.reshape(tf.concat([a, b_tile], -1), [-1, a_dim+b_dim])  #(b*m, d_a+d_b)

        a_re = a_re + tf.nn.relu(tf.matmul(ab_fusion, W)+ bias)  #(b*m, d_a)

        output = tf.concat([tf.matmul(
                            tf.reshape(tf.matmul(a_re, T_k), [-1, element_size, emb_dim]),
                            tf.expand_dims(b, -1))
                            for T_k in tf.unstack(T_dropout, axis=0)], -1)
        if activation != None:
            output = activation(output)

        return output