"""
@Function:  用于判断改块用传统方法的效果好还是非传统方法的效果好
1： 使用网络的效果更好
2： 不使用网络的效果更好
@Author : ZhangPeiCheng
@Time : 2024/1/6 10:40
"""
import keras.backend
import numpy as np
from keras import Model, Input
from keras.layers import Dense, Flatten, BatchNormalization, \
    MultiHeadAttention, Reshape, concatenate, Conv1D, MaxPooling1D, Dropout, Masking, LSTM, Add, Concatenate, Lambda, \
    Conv2D
import tensorflow as tf

from model.predict.predict_v2_conv.components.get_decoder_input import MaskToken, MaskToken2
from model.predict.predict_v2_conv.components.get_token import TokenGenerator, pos_encode
from model.predict.predict_v2_conv.components.get_transformer_block import TransformerEncoderBlock, \
    TransformerDecoderBlock
from model.predict.predict_v2_conv.components.get_feature_extractor import FeatureExtractor
from model.predict.predict_v2_conv.components.get_pixels_seq import SeqRefresh


def residual_block(block, stride):
    pic = Masking(mask_value=-1)(block)
    # attention, pic shape is (None, width ,length , dim)
    x = Conv2D(kernel_size=(3, 3), filters=16, padding='same', strides=1)(pic)
    x = BatchNormalization()(x)
    x = Conv2D(kernel_size=(3, 3), filters=16, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    # print(x.shape)
    # residual_connect = Conv2D(kernel_size=(1, 1), filters=filter, padding='same', strides=1)(temp)
    return x


def residual_block_(block, stride):
    pic = Masking(mask_value=-1)(block)
    # attention, pic shape is (None, width ,length , dim)
    x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', strides=1)(pic)
    x = BatchNormalization()(x)
    x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', strides=1)(x)
    x = BatchNormalization()(x)
    # print(x.shape)
    # residual_connect = Conv2D(kernel_size=(1, 1), filters=filter, padding='same', strides=1)(temp)
    return x


def extract_features(block, stride=2, residual_block_cnt=1):
    """
    输入一张图片得到卷积之后的特征
    由于图像块比较小，这个地方就简单加上一个残差块来进行特征的提取
    :return:
    """
    # masking the value of decoder output
    block = Masking(mask_value=-1)(block)
    # attention, pic shape is (None, width ,length , dim)
    for i in range(residual_block_cnt):
        # 暂时先用3作为kernel_size
        block = residual_block(block, stride)
        block_ = residual_block_(block, stride)
        block = Concatenate(axis=1)([block, block_])
    return block


def forward_pass(feature_input):
    flatten_vector = Flatten()(feature_input)
    x = Dense(units=16)(flatten_vector)
    x = Dense(units=8)(x)
    x = Dense(units=4)(x)
    x = Dense(units=2, activation='softmax')(x)
    return x


def classifier(block_shape):
    block_input = Input(block_shape)
    # 添加卷积所需要的通道
    block_input = Reshape((block_input.shape[1], block_input.shape[2], 1))(block_input)
    feature_vectors = extract_features(block_input, stride=2, residual_block_cnt=2)
    result = forward_pass(feature_vectors)
    # 构建模型
    model = Model(inputs=block_input, outputs=result)
    return model


if __name__ == "__main__":
    # 暂时先以首个元素为unmask来构建网络
    model = classifier((8, 8))
    model.summary()
