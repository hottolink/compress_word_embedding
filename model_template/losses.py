#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, sys, io
from keras import backend as K

# simple squared loss
def squared_error(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true), axis=-1)