"""
@Function:  Initialization of token
@Author : ZhangPeiCheng
@Time : 2023/12/2 18:35
"""
import tensorflow as tf
import numpy as np
from keras import layers


# positional encoding
def pos_encode(pos, d_model):
    def get_angle(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angle(np.arange(pos)[:, np.newaxis],
                           np.arange(d_model)[np.newaxis, :],
                           d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


# reference: https://keras.io/examples/vision/image_classification_with_vision_transformer/
# edited positional encoding part

class TokenGenerator(layers.Layer):
    def __init__(self, num_pixel, projection_dim, *args, **kwargs):
        super(TokenGenerator, self).__init__(*args, **kwargs)
        self.num_pixel = num_pixel
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = pos_encode(
            pos=num_pixel, d_model=projection_dim
        )

    def call(self, patch):
        # Projection and join position coding
        encoded = self.projection(patch) + self.position_embedding
        return encoded

    def get_config(self):
        config = {"num_pixel": self.num_pixel,
                  "projection_dim": self.projection_dim}
        return config

    @classmethod
    def from_config(cls, config):
        # temp
        return cls(**config)
