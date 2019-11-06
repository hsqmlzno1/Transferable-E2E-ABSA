
import os
import sys
import time
import random
from tqdm import tqdm
import numpy as np
import math
import tensorflow as tf

sys.path.insert(0, 'models')

from config import *
from utils import *
from evals import *
from models import Trans_E2E_ABSA
from accumulator import Accumulator

tf.set_random_seed(args.random_seed)
np.random.seed(args.random_seed)

separator = '========================================================================================'

def run(sess, batch_gen, dataset, model, params):

    S_batches, T_batches = batch_gen
    train_set, val_set, tar_un_set, test_set = dataset

    n_train = len(train_set['lm_labels'])
    batch_num = n_train / params.batch_size
    print('training number:%d, batch num:%d' % (n_train, batch_num))

    best_val_ote_score, best_val_ts_score = -999.0, -999.0
    best_epoch = -1

    start_time = time.time()
    losses = Accumulator(['loss', 'asp_loss', 'ts_loss', 'opn_loss','ote_transfer_loss'], batch_num)

    for epoch in range(params.n_epoch):

        cur_lr = params.lr

        for i in range(batch_num):

            xs, win_xs, length_s, ys_ote, ys_ts, ys_opn, ys_stm, _, _ = S_batches.next()
            xt, win_xt, length_t, yt_ote, yt_ts, yt_opn, yt_stm, _, _ = T_batches.next()
            x = np.vstack([xs, xt])
            win_x  = np.vstack([win_xs, win_xt])
            length = np.hstack([length_s, length_t])
            y_ote  = np.vstack([ys_ote, yt_ote])
            y_ts   = np.vstack([ys_ts, yt_ts])
            y_opn   = np.vstack([ys_opn, yt_opn])
            y_stm  = np.vstack([ys_stm, yt_stm])

            feed_dict = get_train_feed_dict(model, x, win_x, length, y_ote, y_ts, y_opn, y_stm, cur_lr, params.dropout_rate, train_flag=True)

            _, loss, asp_loss, ts_loss, opn_loss = sess.run([model.train_op, model.loss, model.asp_loss, model.ts_loss, model.opn_loss], feed_dict=feed_dict)
            _, ote_transfer_loss = sess.run([model.ote_transfer_op, model.ote_transfer_loss], feed_dict=feed_dict)
            losses.add([loss, asp_loss, ts_loss, opn_loss, ote_transfer_loss])

        if epoch % params.evaluation_interval == 0:

            print('--------------------epoch %d--------------------' % (epoch+1))
            print('learning_rate:', cur_lr)

            losses.output('time %.5fs,' % (time.time() - start_time))
            losses.clear()

            train_ote_scores, train_ts_scores, _, _ = eval_metric(sess, model, params, train_set, domain_flag=True)
            train_ote_p, train_ote_r, train_ote_f1 = train_ote_scores
            train_ts_macro_f1, train_ts_micro_p, train_ts_micro_r, train_ts_micro_f1 = train_ts_scores

            print("train performance: ote: precision: %.4f, recall: %.4f, f1: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f"
                  % (train_ote_p, train_ote_r, train_ote_f1, train_ts_micro_p, train_ts_micro_r, train_ts_micro_f1))

            val_ote_scores, val_ts_scores, _, _ = eval_metric(sess, model, params, val_set, domain_flag=True)
            val_ote_p, val_ote_r, val_ote_f1 = val_ote_scores
            val_ts_macro_f1, val_ts_micro_p, val_ts_micro_r, val_ts_micro_f1 = val_ts_scores

            print("val performance: ote: precision: %.4f, recall: %.4f, f1: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f"
                  % (val_ote_p, val_ote_r, val_ote_f1, val_ts_micro_p, val_ts_micro_r, val_ts_micro_f1))

            if args.selection_schema == 'OTE_TS':
                if val_ts_micro_f1 > best_val_ts_score and val_ote_f1 > best_val_ote_score:
                    best_val_ts_score = val_ts_micro_f1
                    best_val_ote_score = val_ote_f1
                    best_epoch = epoch + 1
                    print("Save...")
                    model.save_model(sess)
            if args.selection_schema == 'TS':
                if val_ts_micro_f1 > best_val_ts_score:
                    best_val_ts_score = val_ts_micro_f1
                    best_val_ote_score = val_ote_f1
                    best_epoch = epoch + 1
                    print("Save...")
                    model.save_model(sess)

    print('Store the best model at the epoch: %d\n' % best_epoch)

if __name__ == '__main__':

    # build dataset
    train, val, tar_un, test, vocab, char_vocab, ote_tag_vocab, ts_tag_vocab = build_dataset(
        source_domain=args.source_domain, target_domain=args.target_domain, input_win=args.input_win,
        tagging_schema=args.tagging_schema, stm_win=args.stm_win
    )
    args.max_len = len(train[0]['lm_labels'])

    # obtain the pre-trained word embeddings
    emb_name = args.emb_name
    emb_path = emb2path[emb_name]
    embeddings = load_embeddings(path=emb_path, vocab=vocab, source_domain=args.source_domain, target_domain=args.target_domain, emb_name=emb_name)

    args.dim_w = len(embeddings[0])
    args.ote_tag_vocab = ote_tag_vocab
    args.ts_tag_vocab = ts_tag_vocab
    args.dim_ote_y = len(ote_tag_vocab)
    args.dim_ts_y = len(ts_tag_vocab)
    args.n_epoch = pair2epoch[args.source_domain+'-'+args.target_domain]
    args.selection_schema = pair2schema[args.source_domain+'-'+args.target_domain]

    print(separator)
    for arg in vars(args):
        arg_string = "\t-%s: %s" % (arg, str(getattr(args, arg)))
        print(arg_string)

    log_dir = './work/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.model_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train = transform_data_format(train)
    val = transform_data_format(val)
    tar_un = transform_data_format(tar_un)
    test = transform_data_format(test)

    S_batches = batch_generator(train, args.batch_size, shuffle=True)
    T_batches = batch_generator(tar_un, args.batch_size, shuffle=True)

    gpu_config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=gpu_config) as sess:
        model = Trans_E2E_ABSA(args, word_vecs=embeddings, name='Trans_E2E_ABSA')
        model.initialize_session(sess, args.round)

        if args.train:

            run(sess=sess,
                batch_gen=[S_batches, T_batches],
                dataset=[train, val, tar_un, test],
                model=model,
                params=args)

        if args.test:

            model.load_model(sess)

            test_ote_scores, test_ts_scores, test_pred_ote, test_pred_ts = eval_metric(sess, model, args, test, domain_flag=False)
            test_ote_p, test_ote_r, test_ote_f1 = test_ote_scores
            test_ts_macro_f1, test_ts_micro_p, test_ts_micro_r, test_ts_micro_f1 = test_ts_scores

            print("test performance: ote: precision: %.4f, recall: %.4f, f1: %.4f, ts: precision: %.4f, recall: %.4f, micro-f1: %.4f"
                  % (test_ote_p, test_ote_r, test_ote_f1, test_ts_micro_p, test_ts_micro_r, test_ts_micro_f1))

            store_results(args.model_name, args.round, args.source_domain, args.target_domain,
                          test_ote_scores, test_ts_scores)

            store_predictions(args, test, test_pred_ote, test_pred_ts, test_ts_micro_f1)
