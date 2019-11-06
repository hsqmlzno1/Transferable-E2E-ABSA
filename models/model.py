
from __future__ import absolute_import
from __future__ import division

import os
from functools import partial
import tensorflow as tf

import architectures
import nn_utils
from flip_gradient import flip_gradient
from utils import *

class Trans_E2E_ABSA(object):

    def __init__(self,
                 args,
                 word_vecs,
                 init=tf.random_uniform_initializer(minval=-0.2, maxval=0.2),
                 name='Trans_TBSA'):

        self.args = args
        self.word_vecs = word_vecs
        self.init = init
        self.name = name

        self.batch_size = args.batch_size
        self.max_len = args.max_len

        self.dim_w = args.dim_w
        self.dim_asp_h = args.dim_asp_h
        self.dim_opn_h = args.dim_opn_h
        self.dim_ts_h = args.dim_ts_h
        self.dim_rel = args.dim_rel
        self.dim_ote_y = args.dim_ote_y
        self.dim_ts_y = args.dim_ts_y
        self.dim_lm_y = args.dim_lm_y
        self.input_win = args.input_win
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.clip_grad = args.clip_grad
        self.batch_size = args.batch_size

        self.build_inputs()
        self.build_vars()
        self.build_graph()


    def build_inputs(self):

        with tf.name_scope('input'):

            self.reviews = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="reviews")
            self.win_reviews = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len, self.input_win], name="win_reviews")
            self.batch_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name="batch_length")

            self.asp_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="asp_labels")
            self.opn_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="opn_labels")
            self.ts_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="ts_labels")
            self.stm_lm_labels = tf.placeholder(dtype=tf.int32, shape=[None, self.max_len], name="stm_lm_labels")

            self.lr = tf.placeholder(dtype=tf.float32, shape=[], name="learning_rate")
            self.adapt = tf.placeholder(dtype=tf.float32, shape=[], name="adapt_rate")
            self.dropout_rate = tf.placeholder_with_default(1., shape=())
            self.train_flag = tf.placeholder(dtype=tf.bool)
            self.domain_flag = tf.placeholder(dtype=tf.bool)

    def build_vars(self):

        with tf.variable_scope(self.name):

            word_vecs = tf.convert_to_tensor(self.word_vecs)
            self.word2vec = tf.Variable(word_vecs, name="word2vec", trainable=True)
            self.nil_vars = set([self.word2vec.name])

            # global memories for aspect and opinion
            self.ma = tf.Variable(self.init([1, 2*self.dim_asp_h]), name='ma')
            self.mo = tf.Variable(self.init([1, 2*self.dim_opn_h]), name='mo')

            self.SuperNet = architectures.SuperNN(self.args, self.word2vec, init=self.init, scope="shared")

    def build_graph(self):

        with tf.variable_scope(self.name):

            src_reviews = tf.cond(self.train_flag, lambda: tf.slice(self.reviews, [0, 0], [self.batch_size, self.max_len]), lambda: self.reviews)
            tar_reviews = tf.cond(self.train_flag, lambda: tf.slice(self.reviews, [self.batch_size, 0], [self.batch_size, self.max_len]), lambda: self.reviews)
            src_batch_length = tf.cond(self.train_flag, lambda: tf.slice(self.batch_length, [0], [self.batch_size]), lambda: self.batch_length)
            tar_batch_length = tf.cond(self.train_flag, lambda: tf.slice(self.batch_length, [self.batch_size], [self.batch_size]), lambda: self.batch_length)

            asp_labels = tf.cond(self.train_flag, lambda: tf.slice(self.asp_labels, [0, 0], [self.batch_size, self.max_len]), lambda: self.asp_labels)
            ts_labels  = tf.cond(self.train_flag, lambda: tf.slice(self.ts_labels,  [0, 0], [self.batch_size, self.max_len]),  lambda: self.ts_labels)
            asp_labels = tf.reshape(asp_labels, [-1])
            ts_labels  = tf.reshape(ts_labels, [-1])
            opn_labels = tf.reshape(self.opn_labels, [-1])

            src_weights_mask = tf.cast(tf.sign(src_reviews), tf.float32)
            weights_mask = tf.cast(tf.sign(self.reviews), tf.float32)

            asp_h_s, ts_h_s, asp_pred_s, opn_pred_s, ts_pred_s, ma_s, mo_s, alpha_a_s, alpha_o_s = self.SuperNet(src_reviews, src_batch_length, self.ma, self.mo, self.dropout_rate, reuse=False)
            asp_h_t, ts_h_t, asp_pred_t, opn_pred_t, ts_pred_t, ma_t, mo_t, alpha_a_t, alpha_o_t = self.SuperNet(tar_reviews, tar_batch_length, self.ma, self.mo, self.dropout_rate, reuse=True)
            opn_pred = tf.concat([opn_pred_s, opn_pred_t], 0)

            asp_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(asp_labels, self.dim_ote_y), logits=asp_pred_s, name='ote_tagger') #(b*m, dim_ote_y)
            opn_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(opn_labels, self.dim_lm_y), logits=opn_pred, name='opn_tagger') #(b*m, dim_lm_y)
            ts_loss  = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(ts_labels, self.dim_ts_y), logits=ts_pred_s,  name='ts_tagger')  #(b*m, dim_ts_y)

            asp_attention = tf.concat([alpha_a_s, alpha_a_t], 0)
            self.ote_transfer_loss = self.add_adv_loss(asp_h_s, asp_h_t,
                                                       adapt_rate=self.args.adapt_rate,
                                                       attention=asp_attention,
                                                       mask=weights_mask,
                                                       batch_length=self.batch_length,
                                                       selective=self.args.selective,
                                                       weight=self.args.adv_weight, scope='asp_dann')

            asp_loss = tf.reshape(asp_loss, [-1, self.max_len]) * src_weights_mask  #(b, m)
            ts_loss  = tf.reshape(ts_loss,  [-1, self.max_len]) * src_weights_mask  #(b, m)
            opn_loss = tf.reshape(opn_loss, [-1, self.max_len]) * weights_mask  #(b, m)

            asp_loss = tf.reduce_sum(asp_loss, axis=-1) / tf.cast(src_batch_length, tf.float32) #(b)
            ts_loss  = tf.reduce_sum(ts_loss,  axis=-1) / tf.cast(src_batch_length, tf.float32) #(b)
            opn_loss = tf.reduce_sum(opn_loss, axis=-1) / tf.cast(self.batch_length, tf.float32) #(b)

            self.asp_loss = tf.reduce_mean(asp_loss)
            self.ts_loss  = tf.reduce_mean(ts_loss)
            self.opn_loss = tf.reduce_mean(opn_loss)

            self.loss = self.asp_loss + self.opn_loss + self.ts_loss

            asp_pred = tf.cond(self.domain_flag, lambda: asp_pred_s, lambda: asp_pred_t)
            ts_pred = tf.cond(self.domain_flag, lambda: ts_pred_s, lambda: ts_pred_t)
            self.asp_attentions = tf.cond(self.domain_flag, lambda: alpha_a_s, lambda: alpha_a_t)
            self.opn_attentions = tf.cond(self.domain_flag, lambda: alpha_o_s, lambda: alpha_o_t)

            self.asp_predictions = tf.reshape(tf.argmax(asp_pred, -1, name="asp_predictions"), [-1, self.max_len])
            self.ts_predictions  = tf.reshape(tf.argmax(ts_pred, -1, name="ts_predictions"), [-1, self.max_len])

            # determine the optimizer
            if self.args.optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif self.args.optimizer == "momentum":
                self.opt = tf.train.MomentumOptimizer(self.lr, 0.9)
            elif self.args.optimizer == "adadelta":
                self.opt = tf.train.AdadeltaOptimizer(self.lr)
            elif self.args.optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            else:
                raise Exception("Unsupported optimizer type: %s" % self.args.optimizer)

            var_list     = [tf_var for tf_var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]

            self.train_op  = nn_utils.train_network(self.opt, self.loss, var_list, self.nil_vars, self.clip_grad, "train_op")
            self.ote_transfer_op  = nn_utils.train_network(self.opt, self.ote_transfer_loss, var_list, self.nil_vars, self.clip_grad, "ote_dann_op")

    def add_adv_loss(self, src_feat, tar_feat, adapt_rate, attention, mask, batch_length, selective=True, weight=1.0, scope=None):

        dom_label = tf.concat([tf.tile(tf.zeros(1, dtype=tf.int32), [tf.shape(src_feat)[0]]),
                               tf.tile(tf.ones(1, dtype=tf.int32),  [tf.shape(tar_feat)[0]])], 0)

        feat = tf.concat([src_feat, tar_feat], 0)
        feat = flip_gradient(feat, adapt_rate)

        dom_fc = nn_utils.fc_layer(feat, output_dim=feat.shape.as_list()[-1], scope=scope+'/fc1', reuse=False)
        dom_logit = nn_utils.fc_layer(dom_fc, output_dim=2, scope=scope+'/fc2', reuse=False)
        domain_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(dom_label, 2), logits=dom_logit, name='dom_classifier')  # (b*m, )
        domain_loss = tf.reshape(domain_loss, [-1, self.max_len]) * mask  # (b, m)

        if selective:
            print('selective adversarial loss')
            domain_loss = tf.reduce_sum(domain_loss*attention, axis=-1) / tf.cast(batch_length, tf.float32)  # (b)
        else:
            print('adversarial loss')
            domain_loss = tf.reduce_sum(domain_loss, axis=-1) / tf.cast(batch_length, tf.float32)  # (b)
        domain_loss = weight*tf.reduce_mean(domain_loss)

        return domain_loss

    def initialize_session(self, sess, round):

        sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        model_dir = "./work/snapshot/"
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        model_dir = model_dir+"%s/" % self.args.model_name
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        output_dir = model_dir+"%s-%s/" %(self.args.source_domain, self.args.target_domain)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        self.save_path= output_dir + 'Round%d.ckpt' % round

    def save_model(self, sess):
        self.saver.save(sess, self.save_path)

    def load_model(self, sess):
        try:
            self.saver.restore(sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model " "from save path: %s" % self.save_path)
        self.saver.restore(sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)