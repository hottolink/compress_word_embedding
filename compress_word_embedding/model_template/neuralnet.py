#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
from builtins import range

from keras import backend as K
from keras.layers import Dense, Add, Input, Lambda, Activation
from keras.models import Model
from .noise_layers import GumbelNoise

def compress_word_embedding(N_k, N_m, N_dim, N_h=None):
    """
    (Denoising) Auto-Encoder for compressing various embeddings, i.e. Word Embedding
    :param N_k: number of codeword vectors
    :param N_m: number of codebooks
    :param N_dim: dimension size of original embedding
    :param N_h: dimension size of hidden layer. default:N_k*N_m/2
    :return: keras model proto
    """
    # type: (int, int, int, int) -> (keras.models.Model)
    N_h = int(N_k*N_m//2) if N_h is None else N_h

    # create instance of functional layer
    log_layer = Lambda(lambda x: K.log(x), name="log_layer")
    softmax_layer = Activation("softmax", name="gumbel_softmax")

    # create input variables
    vec_x = Input(shape=(N_dim,), dtype="float32", name="input_x")
    lst_gumbel_noise = [Input(shape=(N_k,), dtype="float32", name="input_g_%d" % i) for i in range(N_m)]
    lst_input = [vec_x]
    lst_input.extend(lst_gumbel_noise)

    # Encoder:x -> h -> {a_i} -> {d_i}
    ## hidden layer
    lst_A_enc = [Dense(units=N_k, activation=K.softplus, use_bias=True, name="h_to_d_dash_%d" % i) for i in range(N_m)]
    h = Dense(units=N_h, activation=K.tanh, use_bias=True, name="x_to_h")(vec_x)
    lst_d_dash = [A_enc(h) for A_enc in lst_A_enc]
    ## apply logarithm
    lst_d_dash = [log_layer(d_dash) for d_dash in lst_d_dash]
    ## add noise
    add_noise_layer = Add(name="add_noise")
    lst_d_dash = [add_noise_layer([d, g]) for d, g in zip(lst_d_dash, lst_gumbel_noise)]
    ## apply softmax: {d_i}
    lst_d = [softmax_layer(d_dash) for d_dash in lst_d_dash]
    # Decoder: {d_i} -> {x'_i} -> x'
    lst_v_dec = [Dense(units=N_dim, activation=None, use_bias=False, name="decoder_%d" % i )(d) for i, d in enumerate(lst_d)]
    ### sum vectors
    v_dec = Add(name="decoder_add")(lst_v_dec)

    # construct model proto
    model = Model(inputs=lst_input, outputs=[v_dec])

    return model


def compress_word_embedding_simple(N_k, N_m, N_dim, F_temperature, N_h=None):
    """
    (Denoising) Auto-Encoder for compressing various embeddings, i.e. Word Embedding
    :param N_k: number of codeword vectors
    :param N_m: number of codebooks
    :param N_dim: dimension size of original embedding
    :param F_temperature: scale parameter for gumbel noise
    :param N_h: dimension size of hidden layer. default:N_k*N_m/2
    :return: keras model proto
    """
    # type: (int, int, int, float, int) -> (keras.models.Model)

    N_h = int(N_k*N_m//2) if N_h is None else N_h

    # create instance of functional layer
    log_layer = Lambda(lambda x: K.log(x), name="log_layer")
    softmax_layer = Activation("softmax", name="gumbel_softmax")

    # create input variables
    vec_x = Input(shape=(N_dim,), dtype="float32", name="input_x")

    # Encoder:x -> h -> {a_i} -> {d_i}
    ## hidden layer
    lst_A_enc = [Dense(units=N_k, activation=K.softplus, use_bias=True, name="h_to_d_dash_%d" % i) for i in range(N_m)]
    h = Dense(units=N_h, activation=K.tanh, use_bias=True, name="x_to_h")(vec_x)
    lst_d_dash = [A_enc(h) for A_enc in lst_A_enc]
    ## apply logarithm
    lst_d_dash = [log_layer(d_dash) for d_dash in lst_d_dash]
    ## add noise
    add_noise_layer = GumbelNoise(scale=F_temperature, name="add_noise")
    lst_d_dash = [add_noise_layer(d_dash) for d_dash in lst_d_dash]
    ## apply softmax: {d_i}
    lst_d = [softmax_layer(d_dash) for d_dash in lst_d_dash]
    # Decoder: {d_i} -> {x'_i} -> x'
    lst_v_dec = [Dense(units=N_dim, activation=None, use_bias=False, name="decoder_%d" % i )(d) for i, d in enumerate(lst_d)]
    ### sum vectors
    vec_x_dec = Add(name="decoder_add")(lst_v_dec)

    # construct model proto
    model = Model(inputs=[vec_x], outputs=[vec_x_dec])

    return model
