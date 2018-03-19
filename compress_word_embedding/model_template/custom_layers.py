#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import keras.backend as K
import tensorflow as tf
from keras.layers import Layer, Lambda
from keras.initializers import glorot_uniform, constant
from keras import regularizers
import numpy as np

class MaskedAveragePooling1D(Layer):

    def __init__(self, input_dim, **kwargs):
        # type: (int, any) -> (MaskedAveragePooling1D)
        """
        Average Pooling layer that supports masked input
        :param kwargs: other arguments
        """
        self.supports_masking = True
        self.trainable = False
        self.input_dim = input_dim
        super(MaskedAveragePooling1D, self).__init__(**kwargs)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def get_config(self):
        config = super(MaskedAveragePooling1D, self).get_config()
        orig_config = {
            "input_dim": self.input_dim}
        config.update(orig_config)
        return config

    def build(self, input_shape):
        super(MaskedAveragePooling1D, self).build(input_shape)

    def call(self, x, mask=None):
        if mask is not None:
            # mask (batch, time, 1)
            mask = K.cast(mask, dtype="float32")
            mask = K.expand_dims(mask)
            # to make the masked values in x be equal to zero
            x *= mask
        return K.sum(x, axis=1) / K.sum(mask, axis=1)

    def compute_output_shape(self, input_shape):
        # remove temporal dimension
        return input_shape[0], input_shape[2]


class PositionalEmbedding(Layer):

    def __init__(self, input_seq_length, input_dim, trainable=False, **kwargs):
        # type: (int, int, bool, any) -> (PositionalEmbedding)
        """
        Encodes positional information into the distributional representation.
        default representation is the sinusoid(=sine curve).
        you can set the representation as trainable, but recent research shows it has negligible impact on the performance.
        ref. Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin.
        Convolutional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.
        :param input_seq_length: maximum input sequence length
        :param input_dim: input vector length
        :param trainable: set positional embedding vector as trainable or not
        :param kwargs:
        """
        self.supports_masking = True

        self.input_seq_length = input_seq_length
        self.input_dim = input_dim
        self.trainable = trainable

        super(PositionalEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        orig_config = {
            "input_seq_length":self.input_seq_length,
            "input_dim":self.input_dim
        }
        config.update(orig_config)
        return config

    def build(self, input_shape):

        # positional_embedding.shape = (time, input_dim)
        mat_pos_embed = self._init_positional_embedding(n_pos=self.input_seq_length, n_dim=self.input_dim)
        self.positional_embedding = self.add_weight(name="positional_embedding", shape=(self.input_seq_length, self.input_dim),
                                         initializer=constant(value=mat_pos_embed), trainable=self.trainable, dtype="float32")
        super(PositionalEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x_in, mask=None):
        # x_in = (batch, time, input_dim)
        if mask is not None:
            # mask(batch, time) -> mask(batch, time, 1)
            mask = K.cast(mask, dtype="float32")
            mask = K.expand_dims(mask, axis=-1)
            # returned = x_in + mask(broadcast) * position_embed
            ret = x_in + mask * K.expand_dims(self.positional_embedding, axis=0)
        else:
            ret = x_in + K.expand_dims(self.positional_embedding, axis=0)
        return ret

    def _init_positional_embedding(self, n_pos, n_dim, normalize=True):
        # mat_pos.shape = (n_pos, n_dim)

        t_max = n_pos / (2*np.pi)
        # u_{n,2i} = n/t_max^(2*i/n_dim); u_{n,2i+1} = n/t_max^(2*i/n_dim)
        mat_pos_embed = np.vstack(pos / np.power(t_max, 2*(np.arange(0, n_dim)//2) / n_dim) for pos in range(n_pos))
        # mat_pos_{n,i} = sin(u_{n,i}) if i % 2 == 0 else cos(u_{n,i})
        mat_pos_embed[:,0::2] = np.sin(mat_pos_embed[:,0::2])
        mat_pos_embed[:,1::2] = np.cos(mat_pos_embed[:,1::2])
        if normalize:
            mat_pos_embed /= np.expand_dims(np.linalg.norm(mat_pos_embed, axis=1), axis=-1)

        return mat_pos_embed



class SparseEmbedding(Layer):

    def __init__(self, n_vocab=None, n_dim=None, combiner=None, **kwargs):
        # type: (int, int, unicode, Any) -> SparseEmbedding
        """
        Embedding layer that supports tf.SparseTensor input
        :param n_vocab: vocabulary size of embedding layer. input word id must lie in [0,n_vocab)
        :param n_dim: length of word vector
        :param combiner: pooling method: one of [mean, sum, sqrtn]
        :param kwargs: other arguments
        """
        self.supports_masking = False
        self.word_vector = None
        if "weights" in kwargs:
            assert isinstance(kwargs["weights"][0], np.ndarray)
            self.word_vector = kwargs["weights"][0]
            self.n_vocab, self.n_dim = self.word_vector.shape
        else:
            assert n_vocab is not None, "you must specify vocabulary size."
            assert n_dim is not None, "you must specify the length of word vector."
            self.n_vocab = n_vocab
            self.n_dim = n_dim

        self.combiner = combiner

        super(SparseEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = super(SparseEmbedding, self).get_config()
        orig_config = {
            "n_vocab":self.n_vocab,
            "n_dim":self.n_dim,
            "combiner":self.combiner
        }
        config.update(orig_config)
        return config

    def build(self, input_shape):
        if self.word_vector is None:
            self.word_vector = self.add_weight(name="word_vector", shape=(self.n_vocab, self.n_dim),
                                               initializer=glorot_uniform(), trainable=True, dtype="float32")
        else:
            self.word_vector = K.variable(self.word_vector)
            self.trainable_weights = [self.word_vector]

        super(SparseEmbedding, self).build(input_shape)

    def call(self, lst_input, mask=None):
        x_id, x_weight = lst_input[0], lst_input[1]
        ret = tf.nn.embedding_lookup_sparse(params=self.word_vector, sp_ids=x_id, sp_weights=x_weight, combiner=self.combiner)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)

class SelfAttention(Layer):

    def __init__(self, input_seq_length, input_dim, hidden_dim, n_att_vector, gamma=1.0, scale_output_layer=False, **kwargs):
        """
        parametric weighted mean pooling as known as Self-Attentive Mechanism
        input tensor with dimension (batch, time, input_dim)
        output tensor with dimension (batch, input_dim * n_att_vector)
        ref. Z. Lin, et al. A Structured Self-attentive Sentence Embedding. arXiv preprint arXiv:1703.03130, 2017.
        :param input_seq_length: maximum input sequence length
        :param input_dim: input vector length
        :param hidden_dim: hidden layer size inside Self-Attentive Mechanism. denoted as d_a in original paper
        :param n_att_vector: number of weight vecotr
        :param gamma: coefficient of loss
        :param scale_output_layer: scale output layer; L2 normalizaton and trainable scaling factor
        :param kwargs:
        """
        self.supports_masking = True
        self.input_seq_length = input_seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_att_vector = n_att_vector
        self.gamma_value = gamma
        self.gamma_scale = 1.0 / np.sqrt(input_seq_length)

        self.batch_identity = np.expand_dims(np.eye(self.n_att_vector, dtype=np.float32), axis=0)
        self.scale_output_layer = scale_output_layer

        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        orig_config = {
            "input_seq_length":self.input_seq_length,
            "input_dim":self.input_dim,
            "hidden_dim":self.hidden_dim,
            "n_att_vector":self.n_att_vector,
            "gamma":self.gamma_value,
            "scale_output_layer":self.scale_output_layer
        }
        config.update(orig_config)
        return config

    def compute_mask(self, inputs, mask=None):
        return None

    def build(self, input_shape):
        # W1_T: (input_dim, hidden_dim)
        self.W1_T = self.add_weight(name="W1_T", shape=(self.input_dim, self.hidden_dim), initializer=glorot_uniform(),
                                    trainable=True, dtype="float32")
        # W2_T: (hidden_dim, n_att_vector)
        self.W2_T = self.add_weight(name="W2_T", shape=(self.hidden_dim, self.n_att_vector), initializer=glorot_uniform(),
                                    trainable=True, dtype="float32")
        # A: (1, n_att_vector, 1)
        if self.scale_output_layer:
            self.A = self.add_weight(name="A", shape=(1, self.n_att_vector, 1), initializer=constant(value=1.0),
                                     trainable=True, dtype="float32")

        super(SelfAttention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        # return: (batch, input_dim * n_att_vector)
        n_mb = input_shape[0]
        return (n_mb, self.input_dim * self.n_att_vector)

    def calc_loss(self, mat_att_norm):
        # mat_loss: (batch, n_att_vector, n_att_vector) = (batch, n_att_vector, time) * (batch, time, n_att_vector)
        mat_loss = K.batch_dot( K.tf.transpose(mat_att_norm, perm=[0,2,1]), mat_att_norm)
        mat_loss -= self.batch_identity
        # loss: (1,)
        # you don't have to preserve batch dimension
        loss = self.gamma_value * K.mean( K.tf.norm(mat_loss, axis=(1,2), ord="fro")**2.0 )
        return loss

    def calc_scaled_loss(self, mat_att_norm, mask):
        """
        this loss function takes input sequence length into account
        :param mat_att_norm:
        """
        # mat_loss: (batch, n_att_vector, n_att_vector) = (batch, n_att_vector, time) * (batch, time, n_att_vector)
        mat_loss = K.batch_dot( K.tf.transpose(mat_att_norm, perm=[0,2,1]), mat_att_norm)
        mat_loss -= self.batch_identity

        # loss_each: (None,)
        loss_each = K.tf.norm(mat_loss, axis=(1,2), ord="fro")**2.0
        # loss_each_coef:  (None,)
        # formula: loss_each_coef[i] := (sqrt(T_i) - 1.0) / sqrt(T_max)
        loss_each_coef = self.gamma_scale * ( K.tf.sqrt( K.tf.reduce_sum(mask, axis=1, keep_dims=False) ) - 1.0 )
        loss = self.gamma_value * K.mean(loss_each * loss_each_coef)
        return loss

    def call(self, x_in, mask=None):

        # x_in: (batch, time, n_dim=input_dim)
        if mask is not None:
            # mask (batch, time; dtype=bool)
            # x_mask: (batch, time, 1; dtype=float32)
            mask = K.cast(mask, dtype="float32")
            x_mask = K.expand_dims(mask, axis=-1)
            # apply mask to input
            x = x_in * x_mask
        else:
            x = x_in

        # mat_x_w1_T: (batch, time, hidden_dim) = (batch, time, input_dim) * (input_dim, hidden_dim)
        mat_x_w1_T = K.dot(x, self.W1_T)
        mat_x_w1_T = K.tanh(mat_x_w1_T) # tanh activation
        # mat_x_w1_T = K.relu(mat_x_w1_T) # ReLU activation
        # mat_x_w1_T_w2_T: (batch, time, n_att_vector) = (batch, time, hidden_dim) * (hidden_dim, n_att_vector)
        mat_x_w1_T_w2_T = K.dot(mat_x_w1_T, self.W2_T)
        # scale mat_att along with tempral dimension so as not to overflow with softmax operation
        mat_x_w1_T_w2_T -= tf.reduce_max(mat_x_w1_T_w2_T, axis=1, keep_dims=True)
        # mat_att[mask] = exp(mat_x_w1_T_w2_T)[mask] == 0 if input is masked
        mat_att = K.exp(mat_x_w1_T_w2_T) * x_mask
        # mat_att_norm: (batch, time, n_att_vector)
        mat_att_norm = mat_att / K.tf.expand_dims( K.sum(mat_att, axis=1) + K.epsilon(), axis=1)
        # mat_att_norm_T: (batch, n_att_vector, time)
        mat_att_norm_T = K.tf.transpose(mat_att_norm, perm=[0,2,1])

        # mat_weighted_x: (batch, n_att_vector, input_dim) = (batch, n_att_vector, time) * (batch, time, input_dim)
        mat_weighted_x = K.batch_dot(mat_att_norm_T, x)

        if self.scale_output_layer:
            # mat_weighted_norm_x: norm_x[b,i,:] = A[i] * x[b,i,:] / ||x[b,i,:]||_L2
            mat_weighted_scaled_x = tf.nn.l2_normalize(mat_weighted_x, dim=-1, epsilon=1e-6) * self.A
            ret = K.batch_flatten(mat_weighted_scaled_x)
        else:
            # ret: (batch, n_att_vector * input_dim)
            # batch_flatten() works as row-major order; ret[b,:] = concat(mat_weighted_x[b,0,:], mat_weighted_x[b,1,:], ...)
            ret = K.batch_flatten(mat_weighted_x)

        # default loss
        # loss = self.calc_loss(mat_att_norm)
        # self.add_loss(loss, x_in)
        # scaled loss
        loss = self.calc_scaled_loss(mat_att_norm, mask)
        self.add_loss(loss, x_in)

        return ret


class MaskedConv1D(Layer):

    def __init__(self, input_dim, kernel_size, n_filter, stride, use_bias=True, **kwargs):
        # type: (int, int, int, int, bool, Any) -> MaskedConv1D
        """
        Temporal Convolution layer that supports masked input
        :param input_dim: input word vector length
        :param kernel_size: size(width) of convolution kernel to temporal direction
        :param n_filter: number of convolution kernel
        :param stride: stride
        :param use_bias: enable kernel specific bias term
        :param kwargs:
        """
        self.supports_masking = True

        self.use_bias = use_bias
        self.kernel_size = kernel_size
        self.n_filter = n_filter
        self.stride = stride
        self.input_dim = input_dim
        self.data_format = "channels_last"
        super(MaskedConv1D, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # mask: (N_mb, T_max)
        # pass-turu masking information: it will be used downstream pooling layer
        return mask

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_size, self.input_dim, self.n_filter),
                                      initializer=glorot_uniform(),
                                      trainable=True,
                                      dtype="float32",
                                      name="kernel")
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.n_filter,),
                                        initializer=glorot_uniform(),
                                        trainable=True,
                                        dtype="float32",
                                        name="bias")

        super(MaskedConv1D, self).build(input_shape)

    def get_config(self):
        config = super(MaskedConv1D, self).get_config()
        orig_config = {
            "use_bias":self.use_bias,
            "kernel_size":self.kernel_size,
            "n_filter":self.n_filter,
            "stride":self.stride,
            "input_dim":self.input_dim
        }
        config.update(orig_config)
        return config

    def compute_output_shape(self, input_shape):
        # output shape: (N_mb, T_max, N_filter)
        return (input_shape[0], input_shape[1], self.n_filter)

    def call(self, x_in, mask=None):
        # x_in: (N_mb, T_max, N_dim=input_dim)
        # x: x_in * mask
        if mask is not None:
            # mask (batch, time; dtype=bool)
            # x_mask: (batch, time, 1; dtype=float32)
            mask = K.cast(mask, dtype="float32")
            x_mask = K.expand_dims(mask, axis=-1)
            # apply mask to input
            x = x_in * x_mask
        else:
            x = x_in

        ret = K.conv1d(x, self.kernel, strides=self.stride, padding="same",
                       data_format=self.data_format, dilation_rate=1)

        if self.use_bias:
            ret = K.bias_add(ret, self.bias, data_format=self.data_format)

        return ret