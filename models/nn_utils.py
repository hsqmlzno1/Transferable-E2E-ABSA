
import tensorflow as tf
import numpy as np
import math

def fc_layer(inputs, output_dim=None, activation=None, scope=None, reuse=False):

    with tf.variable_scope(scope, reuse=reuse):

        _, embed_dim = inputs.shape.as_list()
        W_fc = tf.get_variable(shape=[embed_dim, output_dim], name='weight', dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.2, 0.2))
        b_fc = tf.get_variable(shape=[output_dim], name='bias', dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        if activation != None:
            outputs = activation(tf.matmul(inputs, W_fc) + b_fc)
        else:
            outputs = tf.matmul(inputs, W_fc) + b_fc

    return outputs

def dense(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu, input_type='dense'):
    with tf.name_scope(layer_name):
        weight = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=1. / tf.sqrt(input_dim / 2.)), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]), name='bias')
        if input_type == 'sparse':
            activations = act(tf.sparse_tensor_dense_matmul(input_tensor, weight) + bias)
        else:
            activations = act(tf.matmul(input_tensor, weight) + bias)
        return activations

def mask_softmax(target, axis, mask, epsilon=1e-12, name=None):
    with tf.name_scope(name, 'softmax',[target]):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis) * mask
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / (normalize + epsilon)
        return softmax

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].

    The input Tensor `t` should be a gradient.

    The output will be `t` + gaussian noise.

    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope(name, "add_gradient_noise", [t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(name, "zero_nil_slot", [t]) as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros([1, s])
        return tf.concat([z, tf.slice(t, [1, 0], [-1, -1])], 0, name=name)

def train_network(opt, loss, var_list, nil_vars, max_grad_norm, scope):

    grads_and_vars = opt.compute_gradients(loss, var_list=var_list)
    print(scope)
    for g, v in grads_and_vars:
        if g is not None:
            print(v)
    grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) for g, v in grads_and_vars if g is not None]
    grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
    nil_grads_and_vars = []
    for g, v in grads_and_vars:
        if v.name in nil_vars:
            nil_grads_and_vars.append((zero_nil_slot(g), v))
        else:
            nil_grads_and_vars.append((g, v))
    train_op = opt.apply_gradients(nil_grads_and_vars, name=scope)

    return train_op
