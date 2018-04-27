#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
import argparse
from builtins import range
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TerminateOnNaN
from keras_tqdm import TQDMCallback


wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

from model_template.losses import squared_error
from model_template.neuralnet import compress_word_embedding_simple

### configuration ###
F_temperature = 1.0 # temperature for Gumbel-Softmax Trick
N_minibatch = 128
optimizer_name = "Adam"

cfg_optimizer = {
    "Adam":Adam(lr=0.0001)
}
### end of configuration ###

def _parse_arguments():
    # type: (None) -> (dict[unicode, any])
    parser = argparse.ArgumentParser(description="encoder-decoder model that learns compressed code representation for various embeddings.")
    parser.add_argument("--input_embedding", "-i", required=True, type=unicode, help="input embedding file. it must be NumPy `.npy` format.")
    parser.add_argument("--output_dir", "-d", required=True, type=unicode, help="output directory. output file name will be named after input file name")
    parser.add_argument("--N_k", required=False, type=int, default=16, help="number of codeword vectors(default:16")
    parser.add_argument("--N_m", required=False, type=int, default=64, help="number of codebooks(default:64")
    parser.add_argument("--N_epoch", required=False, type=int, default=10, help="number of epochs(default:10")
    args = parser.parse_args()

    # convert from namespace to dictionary
    dict_args = {arg_name:getattr(args, arg_name) for arg_name in vars(args)}

    # create output file path
    _input_file_name, _ = os.path.splitext(os.path.basename(dict_args["input_embedding"]))
    dict_args["path_out_model"] = os.path.join(dict_args["output_dir"], _input_file_name + ".model")

    dict_args["N_h"] = int(dict_args["N_k"]*dict_args["N_m"]//2)
    dict_args["F_temperature"] = F_temperature

    return dict_args


def main():

    dict_args = _parse_arguments()

    assert os.path.exists(dict_args["input_embedding"]), "specified input file doesn't exist: {input_embedding}".format(**dict_args)
    assert not os.path.exists(dict_args["path_out_model"]), "specified output file already exists: {path_out_model}".format(**dict_args)
    print("input embedding file: {input_embedding}".format(**dict_args))
    print("trained encoder-decoder model will be saved as: {path_out_model}".format(**dict_args))

    print("loading embeddings from file...")
    mat_embedding = np.load(dict_args["input_embedding"])
    N_sample, N_dim = mat_embedding.shape
    print("done. vocabulary: %d, dimension: %d" % (N_sample, N_dim))

    print("build prototype for encoder-decoder model...")
    model = compress_word_embedding_simple(N_dim=N_dim, N_h=dict_args["N_h"], N_k=dict_args["N_k"], N_m=dict_args["N_m"], F_temperature=F_temperature)
    model.summary()

    model.compile(optimizer=cfg_optimizer[optimizer_name], loss=squared_error, metrics=[squared_error])

    progress_callback = TQDMCallback()
    for n_e in range(dict_args["N_epoch"]):
        print("epoch: %d" % n_e)

        model.fit(x=mat_embedding, y=mat_embedding, batch_size=N_minibatch, epochs=1, verbose=0, validation_split=0.01,
                  shuffle=True, callbacks=[progress_callback, TerminateOnNaN()])

        # extract train-loss for n-th iteration
        train_loss = np.mean(progress_callback.running_logs["loss"])
        print("done. squared loss for train set: %f" % train_loss)

    print("finished. saving trained models...: %s" % dict_args["path_out_model"])
    model.save(dict_args["path_out_model"])


if __name__ == "__main__":
    print("Python script for training encoder-decoder model that enables the compression of various embeddings.")
    main()
    print("finished. good-bye.")