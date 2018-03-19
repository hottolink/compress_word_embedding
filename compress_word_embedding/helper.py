#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

from builtins import range
import numpy as np

def sample_gumbel_noise(N_minibatch, N_k, N_m, loc=0., scale=1.0):
    # type: (int, int, int, float, float) -> (list[np.ndarray(float)])
    # ret[m] = {x_bk}; x_bk ~ Gumbel(x;loc,scale)
    ret = [np.random.gumbel(loc=loc, scale=scale, size=N_k*N_minibatch).reshape(N_minibatch, N_k) for _ in range(N_m)]
    return ret

