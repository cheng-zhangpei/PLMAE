"""
@Function: 将图像从卷积转化为像素序列
@Author : ZhangPeiCheng
@Time : 2023/12/10 10:01
"""

import tensorflow as tf
import numpy as np
from keras import layers


@tf.keras.utils.register_keras_serializable()
class SeqRefresh(layers.Layer):
    def __init__(self, *args, **kwargs):
        super(SeqRefresh, self).__init__(*args, **kwargs)

    def bulid(self, input_shape):
        super(SeqRefresh, self).build(input_shape)

    def call(self, inputs):
        height, width, num_channels = inputs.shape[1], inputs.shape[2], 1
        # 构造偶数位置的索引
        even_indices = tf.range(0, width, 2)
        # 构造奇数位置的索引
        odd_indices = tf.range(1, width, 2)
        # print("index:" + str(odd_indices))
        origin_indices = tf.range(0, width, 1)

        extracted_elements = []
        _extracted_elements = [] # 原嵌入位置的像素特征
        for row in range(height):
            # 此处实际上会获得两种不同类型的像素，所以这两种不同的类型的像素，将这两种不同的像素全部抽取出来

            if row % 2 == 1:
                # 对偶数行提取偶数索引的元素
                elements = tf.gather(inputs[:, row, :, :], even_indices, axis=1, batch_dims=0)
            else:
                # 对奇数行提取奇数索引的元素
                elements = tf.gather(inputs[:, row, :, :], odd_indices, axis=1, batch_dims=0)
            # print("测试一行的shape:" + str(elements.shape))
            # 将提取的元素添加到列表中
            # elements = tf.gather(inputs[:, row, :, :], origin_indices, axis=1, batch_dims=0)
            extracted_elements.append(elements)
        merged_vector = tf.concat(extracted_elements, axis=1)
        return merged_vector

    def get_config(self):
        config = {}
        return config

    @classmethod
    def from_config(cls, config):
        # temp
        return cls(**config)
