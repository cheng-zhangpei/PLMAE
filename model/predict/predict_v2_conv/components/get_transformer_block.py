"""
@Function:  define the Transformer Block
@Author : ZhangPeiCheng
@Time : 2023/12/2 18:53
"""
import keras
import tensorflow as tf
import numpy as np
from keras import layers, Sequential, activations
from keras.layers import Dense, MultiHeadAttention

from model.predict.predict_v2_conv.components.get_local_attention import LocalSelfAttention, LocalSelfAttention2
class TransformerEncoderBlock(layers.Layer):
    """Implements a Transformer Encoder block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout,filter, block_size, d_encoder=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_layer = None
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.mlp_block = None
        self.att = None
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.block_size = block_size
        self.d_encoder = d_encoder
        self.filter = filter

    def build(self, input_shape):
        # self.att = layers.MultiHeadAttention(
        #     num_heads=self.num_heads,
        #     # If input_shape[-1] is 512 and num_heads is 8, then the computed key_dim is 64 because 512/8=64. this
        #     # means that the dimension of the query, key, and value of each head is 64 in the multi-head attention
        #     # mechanism.
        #     key_dim=4,
        #     value_dim=4,
        #     # input_shape[-1] = d_model
        #     name="MultiHeadDotProductAttention_12",
        # )
        self.att2 = LocalSelfAttention(
            heads=1,
            size_per_head=4,
            block_size=self.block_size,
            d_encoder=self.d_encoder,
            filter=self.filter,
            name="LocalAttention_12",
        )
        self.att3 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_encoder,
            value_dim=self.d_encoder
        )
        self.mlp_block = Sequential(
            [
                keras.layers.Dense(
                    self.mlp_dim,
                    activation="relu",
                    name="dense_encode_1"
                ),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(input_shape[-1],name= "dense_decode_3"),
                keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = keras.layers.Dropout(self.dropout)

    def call(self, inputs_arr):
        inputs = inputs_arr
        # print("block input shape")
        # print(inputs.shape)
        x = self.att3(inputs,inputs)
        # print("after local attention")
        # print(x.shape)
        x = self.dropout_layer(x)
        x = x + inputs
        y = self.layer_norm2(x)
        y = self.mlp_block(y)
        x = x + y
        x = self.layer_norm1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformerDecoderBlock(layers.Layer):
    """Implements a Transformer Encoder block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, block_size=6, d_decoder=8,filter=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense = None
        self.dropout_layer = None
        self.layer_norm1 = None
        self.layer_norm2 = None
        self.mlp_block = None
        self.att = None
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.block_size = block_size
        self.d_decoder = d_decoder
        self.filter = filter

    def build(self, input_shape):
        # self.att = layers.MultiHeadAttention(
        #     num_heads=self.num_heads,
        #     # If input_shape[-1] is 512 and num_heads is 8, then the computed key_dim is 64 because 512/8=64. this
        #     # means that the dimension of the query, key, and value of each head is 64 in the multi-head attention
        #     # mechanism.
        #     key_dim=4,
        #     value_dim=4,
        #     # input_shape[-1] = d_model
        #     name="MultiHeadDotProductAttention_8",
        # )
        self.att2 = LocalSelfAttention2(
            heads=1,
            size_per_head=4,
            block_size=self.block_size,
            d_decoder=self.d_decoder,
            filter = self.filter,
            name="LocalAttention_2",
        )
        self.att3 = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_decoder,
            value_dim=self.d_decoder
        )
        self.mlp_block = keras.Sequential(
            [
                keras.layers.Dense(
                    self.mlp_dim,
                    activation="relu",
                    name= "dense_decode_1"
                ),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(input_shape[-1],name= "dense_decode_2"),
                keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_8",
        )
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_3"
        )
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_4"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)
        self.dense = Dense(units=self.filter * 2 ,name= "dense_decode_3")

    def call(self, inputs_arr):
        inputs = inputs_arr

        x = self.att3(inputs,inputs)

        # x = self.dense(x)
        x = self.dropout_layer(x)
        # print("before add")
        # print(x.shape)
        # print(inputs.shape)
        # print(x.shape)
        # print(inputs.shape)
        x = x + inputs
        y = self.layer_norm2(x)
        y = self.mlp_block(y)
        x = x + y
        x = self.layer_norm1(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
