"""
@Function: the body of the ste-mae
@Author : ZhangPeiCheng
@Time : 2023/12/3 10:25:43
"""
import keras.backend
import numpy as np
from keras import Model, Input
from keras.layers import Dense, Flatten, BatchNormalization, \
    MultiHeadAttention, Reshape, concatenate, Conv1D, MaxPooling1D, Dropout, Masking, LSTM, Add, Concatenate, Lambda
import tensorflow as tf

from model.predict.predict_v2_conv.components.get_decoder_input import MaskToken, MaskToken2
from model.predict.predict_v2_conv.components.get_token import TokenGenerator, pos_encode
from model.predict.predict_v2_conv.components.get_transformer_block import TransformerEncoderBlock, \
    TransformerDecoderBlock
from model.predict.predict_v2_conv.components.get_feature_extractor import FeatureExtractor
from model.predict.predict_v2_conv.components.get_pixels_seq import SeqRefresh


def mae_with_conv_block(block_shape, spatial_info_shape, mask_indices, un_masked_indices, d_encoder=32,
                        d_decoder=64,
                        dff_encoder=32, dff_decoder=64,
                        num_heads=4, drop=0, N_e=3, N_d=6, filter=24):
    """
     description：
     An easy body of mae: can not import images directly into mae, we should divide the train set first
     example:
     encoder_input: [71,72,71,75,79,...]
     decoder_output:[71,75,72,76,75,79,78,...]
    :param batch_size:
    :param mask_indices: mask value index
    :param un_masked_indices: unmask value index
    :param encoder_input_shape: encoder input shape
    :param d_encoder: the token dim, the cnt of linear layer unit
    :param d_decoder: token dim of decoder input
    :param dff_encoder: the mlp dim of the encoder
    :param dff_decoder: the mlp dim of the decoder
    :param num_heads: the num of head of multi-head-attention
    :param drop: drop out rate
    :param N_e:  the number of encoders
    :param N_d:  the number of decoders
    :param filter:  the number of filter in conv
    :return: the output of the model
    """
    # [seg]-------------------------------------------token builder-------------------------------------------

    block = Input(block_shape)
    spatial_input = Input(spatial_info_shape)
    # get feature map ->
    feature_map = FeatureExtractor(block, filter)
    seqRefresh = SeqRefresh()
    seq = seqRefresh(feature_map)  # seq_是
    spatial_info_input = Dense(units=int(d_decoder / 2))(spatial_input)
    tokens = TokenGenerator(seq.shape[1], projection_dim=d_encoder)(seq)
    x = tokens
    # [seg]------------------------------------------encoder builder------------------------------------------
    for _ in range(N_e):
        x = TransformerEncoderBlock(num_heads=num_heads, filter=filter, mlp_dim=dff_encoder, dropout=drop,
                                    block_size=block_shape[0])(x)
    encoder_output = x
    mask_tool = MaskToken2(mask_indices, un_masked_indices)
    encoder_output = Dense(units=d_decoder)(encoder_output)
    decoder_input = mask_tool(encoder_output, spatial_info_input)
    # # [seg]--------------------------------------- Positional Embedding---------------------------------------
    x = decoder_input + pos_encode(decoder_input.shape[1], d_decoder)
    x = Dense(units=d_decoder)(x)
    # # [seg]------------------------------------------decoder builder------------------------------------------
    for _ in range(N_d):
        x = TransformerDecoderBlock(num_heads=num_heads, mlp_dim=dff_decoder, dropout=drop, block_size=block_shape[0])(
            x)
    decoder_output = x
    extracted_tensor = tf.gather(decoder_output, mask_indices, axis=1)  # get masked position
    output = Dense(units=block_shape[2])(extracted_tensor)
    model = Model(inputs=[block, spatial_input], outputs=output)
    return model


# 下面是网络测试部分
if __name__ == "__main__":
    # 暂时先以首个元素为unmask来构建网络
    block_index_limit = 6 * 6
    unmask_index = np.arange(0, block_index_limit, 2)
    mask_index = np.arange(1, block_index_limit, 2)
    unmask_index = tf.convert_to_tensor(unmask_index)
    mask_index = tf.convert_to_tensor(mask_index)
    mae_with_conv_block((6, 6, 1), (18, 4), mask_index, unmask_index, )
    # model.summary()
