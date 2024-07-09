"""
@Function:  get decoder input
Tentatively follow the checkerboard grid for mask insertion
@Author : ZhangPeiCheng
@Time : 2023/12/2 18:58
"""

import tensorflow as tf
import numpy as np
from keras.layers import Dense, Masking
from tensorflow.python.keras.engine.base_layer_v1 import Layer
from tensorflow.python.keras.layers import Concatenate


class MaskToken(Layer):
    """Append a mask token to encoder output."""

    def __init__(self, mask_indices, un_masked_indices, *args, **kwargs):
        super(MaskToken, self).__init__(*args, **kwargs)
        self.mask_indices = mask_indices
        self.un_masked_indices = un_masked_indices
        self.indices = None
        self.mst = None
        self.hidden_size = None

    def build(self, input_shape):
        self.hidden_size = input_shape[-1]
        # The model can learn what information is more appropriate for the task by inserting it at the "mask"
        # position. By associating a trainable variable with the "mask", the model can dynamically learn how to use
        # this information instead of simply using a fixed token. This trainable nature makes the model more adaptable
        self.mst = tf.Variable(
            name="mst",
            initial_value=tf.random.normal(
                shape=(1, 1, self.hidden_size), dtype='float32'),
            trainable=True
        )

    def call(self, input_array):
        inputs = input_array
        # print("掩码 index shape 测试")
        # print(self.mask_indices.shape)
        # print(self.un_masked_indices.shape)
        # print("----------------------------------")
        batch_size = tf.shape(inputs)[0]
        mask_num = self.mask_indices.shape[0]
        # broadcast mask token for batch
        mst_broadcast = tf.cast(
            tf.broadcast_to(self.mst, [batch_size, mask_num, self.hidden_size]),
            dtype=inputs.dtype,
        )
        # print("掩码broadcast测试")
        # print(mst_broadcast.shape)
        # print("----------------------------------")
        # concat
        self.indices = tf.concat([self.mask_indices, self.un_masked_indices], axis=0)
        updates = tf.concat([mst_broadcast, inputs], axis=1)
        # Now index and update are one-to-one correspondences
        # example:
        # [2,4,6,8,10...1,3,5,7,9...,]
        # [mask,mask,mask,mask,mask...71,72,74,70,74...]
        out = tf.gather(updates, self.indices, axis=1, batch_dims=0)
        # print("掩码out测试")
        # print(out.shape)
        # print("----------------------------------")
        return out

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MaskToken2(Layer):
    """Append a mask token to encoder output."""

    def __init__(self, mask_indices, un_masked_indices, *args, **kwargs):
        super(MaskToken2, self).__init__(*args, **kwargs)
        self.batch_size = None
        self.dense = None
        self.mask_indices = mask_indices
        self.un_masked_indices = un_masked_indices
        self.indices = None
        self.mst = None
        self.hidden_size = None

    def build(self, input_shape):
        # print("input_shape")
        # print(input_shape)
        self.dense = Dense(units=8,name="mask2_dense")
        self.dense2 = Dense(units=8,name="mask1_dense")
        self.hidden_size = input_shape[-1]
        # The model can learn what information is more appropriate for the task by inserting it at the "mask"
        # position. By associating a trainable variable with the "mask", the model can dynamically learn how to use
        # this information instead of simply using a fixed token. This trainable nature makes the model more adaptable
        self.mst = tf.Variable(
            name="mst2",
            initial_value=tf.random.normal(
                shape=(self.mask_indices.shape[0], self.hidden_size), dtype='float32'),
            trainable=True
        )
        self.con0 = Concatenate(axis=0,name="con1")
        self.con1 = Concatenate(axis=1,name="con2")
        self.con2 = Concatenate(axis=2,name="con3")
        self.masking = Masking(mask_value=-1)

    def call(self, input_array, sp):
        inputs = input_array
        spatial_information = sp
        # Splicing spatial information with mask information
        # broadcast mask token for batch
        print("--------------mask---------------------")
        print(spatial_information.shape)
        # print(mask_vector.shape)

        expanded_mst = tf.expand_dims(self.mst, axis=0)
        tiled_mst = tf.tile(expanded_mst, [tf.shape(spatial_information)[0], 1, 1])

        # 直接将掩码向量改为外部输入的方式
        mst_broadcast = self.con2([spatial_information, tiled_mst])
        print(mst_broadcast.shape)
        # concat
        self.indices = self.con0([self.mask_indices, self.un_masked_indices])
        # mst_broadcast = self.dense2(mst_broadcast)
        print("----update-----")
        inputs = self.dense(inputs)
        # mst_broadcast = self.masking(mst_broadcast)
        mst_broadcast = self.dense2(mst_broadcast)
        print(inputs.shape)
        updates = self.con1([mst_broadcast,inputs])
        print(updates.shape)

        # Now index and update are one-to-one correspondences
        # example:
        # [2,4,6,8,10...1,3,5,7,9...,]
        # [mask,mask,mask,mask,mask...71,72,74,70,74...]
        out = tf.gather(updates, self.indices, axis=1, batch_dims=0)
        # print("掩码out测试")
        # print(out.shape)
        # print("----------------------------------")
        return out

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
