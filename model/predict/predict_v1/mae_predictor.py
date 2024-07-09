"""
@Function: the body of the ste-mae
@Author : ZhangPeiCheng
@Time : 2023/12/2 13:48
"""
import keras.backend
import numpy as np
from keras import Model, Input
from keras.layers import Dense, Flatten, BatchNormalization, \
    MultiHeadAttention, Reshape, concatenate, Conv1D, MaxPooling1D, Dropout, Masking, LSTM, Add, Concatenate, Lambda
import tensorflow as tf

from model.predict.predict_v1.components.get_decoder_input import MaskToken, MaskToken2
from model.predict.predict_v1.components.get_transformer_block import TransformerEncoderBlock, TransformerEncoderBlock, \
    TransformerDecoderBlock
from model.predict.predict_v1.components.get_token import TokenGenerator, pos_encode


def mae_predictor(encoder_input_shape, mask_indices, un_masked_indices, d_encoder=4,
                  d_decoder=4,
                  dff_encoder=16, dff_decoder=16,
                  num_heads=1, drop=0, N_e=2, N_d=1):
    """
     description：
     An easy body of mae: can not import images directly into mae, we should divide the train set first
     example:
     encoder_input: [71,72,71,75,79,...]
     decoder_output:[71,75,72,76,75,79,78,...]
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
    :return: the output of the model
    """
    # [seg]-------------------------------------------token builder-------------------------------------------

    encoder_inputs = Input(encoder_input_shape)
    tokens = TokenGenerator(encoder_inputs.shape[1], projection_dim=d_encoder)(encoder_inputs)
    x = tokens

    # [seg]------------------------------------------encoder builder------------------------------------------
    for _ in range(N_e):
        x = TransformerEncoderBlock(num_heads=num_heads, mlp_dim=dff_encoder, dropout=drop)(x)
    encoder_output = x
    mask_tool = MaskToken(mask_indices, un_masked_indices)
    decoder_input = mask_tool(encoder_output)
    print("masked encoder output shape")
    print(decoder_input.shape)
    # [seg]--------------------------------------- Positional Embedding---------------------------------------
    x = Lambda(lambda x: x + pos_encode(x.shape[1], d_decoder),name="lambda")(decoder_input)
    x = Dense(units=d_decoder)(x)
    # [seg]------------------------------------------decoder builder------------------------------------------
    for _ in range(N_d):
        x = TransformerDecoderBlock(num_heads=num_heads, mlp_dim=dff_decoder, dropout=drop)(x)
    decoder_output = x
    print("decoder output shape")
    print(decoder_output.shape)
    # extract masked index outputs
    extracted_tensor = tf.gather(decoder_output, mask_indices, axis=1)
    print("extract output shape")
    print(extracted_tensor.shape)
    output = Dense(units=encoder_input_shape[1])(extracted_tensor)
    print("output shape")
    print(output.shape)
    model = Model(inputs=encoder_inputs, outputs=output)
    return model


def mae_with_spatial_info(encoder_input_shape, spatial_info_shape,mask_indices, un_masked_indices, d_encoder=4,
                          d_decoder=8,
                          dff_encoder=16, dff_decoder=72,
                          num_heads=2, drop=0, N_e=3, N_d=6, batch_size=40):
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
    :return: the output of the model
    """
    # [seg]-------------------------------------------token builder-------------------------------------------

    encoder_inputs = Input(encoder_input_shape)
    spatial_info_input = Input(spatial_info_shape)
    # mask_vector = Input(mask_vector_shape)

    # linear projection for spatial information -> half of the dff_encoder -> concat with masked input
    # consider to add more complex operation like self-attention ? or an encoder?
    spatial_info_input = Dense(units=int(d_decoder / 2))(spatial_info_input)
    tokens = TokenGenerator(encoder_inputs.shape[1], projection_dim=d_encoder)(encoder_inputs)
    x = tokens

    # [seg]------------------------------------------encoder builder------------------------------------------
    print("掩码shape")
    print(mask_indices.shape)
    for _ in range(N_e):
        x = TransformerEncoderBlock(num_heads=num_heads, mlp_dim=dff_encoder, dropout=drop)(x)
    encoder_output = x

    mask_tool = MaskToken2(mask_indices, un_masked_indices)
    encoder_output = Dense(units=8)(encoder_output)
    decoder_input = mask_tool(encoder_output, spatial_info_input)
    # [seg]--------------------------------------- Positional Embedding---------------------------------------
    x = decoder_input + pos_encode(decoder_input.shape[1], d_decoder)
    x = Dense(units=d_decoder)(x)
    # [seg]------------------------------------------decoder builder------------------------------------------
    for _ in range(N_d):
        x = TransformerDecoderBlock(num_heads=num_heads, mlp_dim=dff_decoder, dropout=drop)(decoder_input)
    decoder_output = x
    extracted_tensor = tf.gather(decoder_output, mask_indices, axis=1) # get masked position
    print("extract output shape")
    print(extracted_tensor.shape)
    output = Dense(units=encoder_input_shape[1])(extracted_tensor)
    print("output shape")
    print(output.shape)
    model = Model(inputs=[encoder_inputs,spatial_info_input], outputs=output)
    return model


# 下面是网络测试部分
if __name__ == "__main__":
    # 暂时先以首个元素为unmask来构建网络
    block_index_limit = 32 * 32
    unmask_index = np.arange(0, block_index_limit, 2)
    mask_index = np.arange(1, block_index_limit, 2)
    unmask_index = tf.convert_to_tensor(unmask_index)
    mask_index = tf.convert_to_tensor(mask_index)
    model = mae_predictor((32, 1), mask_index, unmask_index, )
    model.summary()
