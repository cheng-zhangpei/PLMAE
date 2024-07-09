"""
@Function:             
@Author : ZhangPeiCheng
@Time : 2024/1/28 20:48
"""

import os
import numpy as np
from keras.saving.save import load_model
from train_set_build.train_conv_set import open_picture, masking_block
import tensorflow as tf

def predict_process(chessboard_blocks, spatial_mask_infos, model_name):
    """加载模型并返回结果
        独立出来不会导致显存冲突
    """
    chessboard_blocks = np.reshape(chessboard_blocks, (chessboard_blocks.shape[0], chessboard_blocks.shape[1],chessboard_blocks.shape[2], 1))
    model = load_model(model_name)
    predict_res = model.predict([chessboard_blocks, spatial_mask_infos])
    return predict_res
def predict_by_model_1(block,model_path):
    """
    :param block: 输入图形块（可输入完整块，masking_block会进行掩码操作）
    :param model_path: 模型路径
    :return: 预测值
    """
    # 获得模型输入输出
    blocks =  []
    blocks.append(block)
    blocks = np.array(blocks)
    encoder_inputs, decoder_outputs, spatial_mask_infos, chessboard_blocks = masking_block(blocks)
    # 预测输入主要两个
    # 1. chessboard_blocks：多个的棋盘格
    # 2. spatial_mask_infos： 棋盘格对应的空间信息
    predict_value = predict_process(chessboard_blocks, spatial_mask_infos,model_path)
    return predict_value.flatten()

def predict_by_model_2(block_path,model_path):
    """
    输入block（单张）所在路径，以及模型路径进行像素预测
    :param block_path: 块路径
    :param model_path: 模型路径
    :return: 预测值
    """
    pic = open_picture(block_path)
    pic = pic.astype(np.int32)
    blocks =  []
    blocks.append(pic)
    blocks = np.array(blocks)
    # 获得模型输入输出
    encoder_inputs, decoder_outputs, spatial_mask_infos, chessboard_blocks = masking_block(blocks)
    # 预测
    predict_value = predict_process(chessboard_blocks, spatial_mask_infos,model_path)
    return predict_value.flatten()

def predict_by_model_3(blocks_directory,model_path):
    """
    输入blocks（多张）所在目录，以及模型路径进行像素预测
    由于目录中图形块比较多，所以暂时没有给实例
    :param blocks_directory: 块路径目录
    :param model_path: 模型路径
    :return: 所有块的预测值
    """
    predict_values = []
    for filename in os.listdir(blocks_directory):
        filepath = os.path.join(blocks_directory, filename)
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        blocks =  []
        blocks.append(pic)
        blocks = np.array(blocks)
        # 获得模型输入输出
        encoder_inputs, decoder_outputs, spatial_mask_infos, chessboard_blocks = masking_block(blocks)
        # 预测
        predict_value = predict_process(chessboard_blocks, spatial_mask_infos,model_path)
        predict_values.append(predict_value)
    return np.array(predict_values).flatten()
gpus = tf.config.experimental.list_physical_devices('GPU')


if __name__ == '__main__':
    # 测试目录中block size=8*8
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # blocks_directory = r"D:\czp\mae\test_model\error_block_10"
    block_path = r"D:\czp\mae\test_model\error_block_10\img-airplane.bmp-pos-(0, 8).bmp"
    # 从测试图形块中挑选一张图片
    test_block = open_picture(block_path)
    test_block = test_block.astype(np.int32)
    # 如果是改block=6 / 10 可以把结尾8改成对应size
    model_path = r"D:\czp\mae\trained_model\index_change_8"
    predict_values_1 = predict_by_model_1(test_block,model_path)
    predict_values_2 = predict_by_model_2(block_path,model_path)
    # predict_values_3 = predict_by_model_3(blocks_directory,model_path)
    print(predict_values_1)
    print(predict_values_2)
    # print(predict_values_3)