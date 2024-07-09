"""
@Function:  基于conv2d的特征抽取器
@Author : ZhangPeiCheng
@Time : 2023/12/7 12:00
"""
import keras.backend
import numpy as np
import six
from keras import Model, Input
from keras.layers import Dense, Flatten, BatchNormalization, \
    MultiHeadAttention, Reshape, concatenate, Conv1D, MaxPooling1D, Dropout, Masking, LSTM, Add, Concatenate, Lambda, \
    Conv2D, Activation
import tensorflow as tf
from keras.regularizers import l2


def FeatureExtractor(pic,filter):
    """
    输入一张图片得到卷积之后的特征
    由于图像块比较小，这个地方就简单加上一个残差块来进行特征的提取
    :return:
    """
    # masking the value of decoder output
    pic = Masking(mask_value=-1)(pic)
    temp = pic
    # attention, pic shape is (None, width ,length , dim)
    x = Conv2D(kernel_size=(3, 3), filters=filter, padding='same', strides=1)(pic)
    x = BatchNormalization()(x)
    x = Conv2D(kernel_size=(5, 5), filters=filter, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    residual_connect = Conv2D(kernel_size=(1, 1), filters=filter, padding='same', strides=1)(temp)
    return x + residual_connect
