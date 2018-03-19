#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from keras.engine import Layer
from keras import backend as K
import numpy as np
from keras.legacy import interfaces


class GumbelNoise(Layer):
    """Apply additive zero-centered Gumbel noise.
    This is useful to implement so-called Gumbel-Softmax Trick[Maddison+ 2016][Jang+ 2016]
    As it is a regularization layer, it is only active at training time.
    # Arguments
        scale: float, scale parameter of the noise distribution, same definition as np.random.gumbel() function
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, scale, **kwargs):
        super(GumbelNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.scale = scale

    def _random_gumbel(self, shape, mean, scale):
        rand_unif = K.random_uniform(shape=shape)
        ret = mean - scale * K.log(-K.log(rand_unif)) # Gumbel(x;mu,scale)
        return ret

    def call(self, inputs, training=None):
        def noised():
            return inputs + self._random_gumbel(shape=K.shape(inputs),
                                            mean=0.,
                                            scale=self.scale)
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'scale': self.scale}
        base_config = super(GumbelNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape