"""
@Function: 测试图像块在不同阈值的错误下，不同波动度指标之间的差距
@Author : ZhangPeiCheng
@Time : 2023/12/16 18:54
"""
import os
import random

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.saving.save import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

from model.train.train_v3_recurrent.temp import liao_method, retraining_block_judge
from test_model.test_conv import generate_binary_stream_for_block, encrypt_chessboard_for_test, random_pixel_index, \
    predict_process, get_residual_result, error_judge, block_res_calc
from train_set_build.sample_sorter import get_SL
from train_set_build.train_conv_set import open_picture, random_block_split, masking_block


def retraining_block_judge_(decrypt_blocks_info, info,wrong_arr):
    """
    Determine if the block needs to be retrained based on threshold
    if the wrong cnt of decrypt process surpass the threshold, the block should be retrained
    """
    info = np.array(info)
    # wrong_index = []  # 记录在解密过程中错误的索引
    for i in range(len(decrypt_blocks_info)):
        decrypt_info = decrypt_blocks_info[i]
        block_origin_info = info[i]
        wrong_cnt = 0
        for j in range(decrypt_info.shape[0]):
            if decrypt_info[j] != block_origin_info[j]:
                wrong_cnt += 1
        wrong_arr[wrong_cnt] += 1
    return wrong_arr
def get_residual_result_(predict_result, encrypt_pixels, origin, bit, info):
    """
    :return:
    """
    bit = 2 ** (bit - 1)
    decrypt_result = []
    for i in range(len(predict_result)):
        # 获得一个block的像素
        encrypt_block_pixel = encrypt_pixels[i]
        decrypt_block_pixel = encrypt_pixels[i]
        predict_block_pixel = predict_result[i]
        origin_block_pixel = origin[i]
        info_block = info[i]
        decrypt_result_block = block_res_calc(predict_block_pixel, encrypt_block_pixel, decrypt_block_pixel,
                                              origin_block_pixel, bit, info_block)
        decrypt_result.append(decrypt_result_block)
    return np.array(decrypt_result)


def blocks_avg_sl(block):
    """
    calc the total SL value for a block
    """
    m, n = block.shape
    Sl_for_block = []
    cnt = 0 # the number of calculate units
    for i in range(m):
        is_odd_row = (i + 1) % 2 == 1
        for j in range(n):
            is_odd_column = (j + 1) % 2 == 1
            if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                SL = get_SL(block,(i, j)) # single SL value for one pixel
                Sl_for_block.append(SL)
                cnt += 1
    return np.sum(Sl_for_block)


def ours_experiment_chessboard(bit, dir_path, pixel_cnt, block_size, model_name):
    """
    棋盘格划分确实有一个比较大的问题就是
    :param bit:
    :param dir_path:
    :param pixel_cnt:
    :param block_size:
    :return:
    """
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        # 打开图片
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        block_cnt = int(pixel_cnt / (block_size * block_size / 2))
        if pixel_cnt % block_size != 0:
            print("需要pixel_cnt可以被block_size整除")
            return
        # 随机选择像素点索引加密
        indexes = random_pixel_index(pixel_cnt)
        # 生成加密信息, 将加密信息只进行简单的切分之后[[block_info1...], [block_info2...], ...]
        info = generate_binary_stream_for_block(block_size)
        # 每个块只有一半的空间可以嵌入信息，暂定第一个像素是嵌入信息的像素
        blocks, block_indexes = random_block_split(pic, block_size, block_cnt)
        # 得到嵌入信息的像素序列以及对于嵌入信息的block
        encrypt_pixels, encrypt_blocks, origin = encrypt_chessboard_for_test(blocks, bit, info)
        # 此处只需要使用encoder_inputs和spatial_mask_infos用于模型的输入
        encoder_inputs, decoder_outputs, spatial_mask_infos,chessboard_blocks = masking_block(encrypt_blocks)
        # 一个图像所有选中图像块的预测结果
        predict_result = predict_process(chessboard_blocks, spatial_mask_infos, model_name)
        # 对预测结果进行reshape
        predict_result = predict_result.reshape(
            (predict_result.shape[0], predict_result.shape[1] * predict_result.shape[2]))
        decrypt_result = get_residual_result_(predict_result, encrypt_pixels, origin, bit, info)
        # 对比嵌入像素和原像素之间的区别
        # 计算SL
        wrong_arr = np.zeros((18, ))
        # 这个解密信息是针对图像的
        wrong_index, wrong_cnt_arr = retraining_block_judge_(decrypt_result, info, wrong_arr)
        error_rate, error_cnt = error_judge(decrypt_result.flatten(), info)
        print(str(filename) + "-错误块数据:"+str(wrong_cnt_arr))
        # 保存 wrong_cnt_arr
        df = pd.DataFrame(wrong_cnt_arr)
        # print(df)
        filename = filename.split(".")[0]
        df.to_csv("./error_cnt_record/"+str(filename)+".csv")
        print("图像:" + str(filename) + "模型错误率:" + str(error_rate) + "," + "错误像素数量: " + str(error_cnt))


import tensorflow as tf
if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    bit = 3
    dir_path = "../test_data"
    # model_name = "../trained_model/test_model_conv_4"
    model_name = "../trained_model/test_model_conv_3"
    print("=========================[seg]===============================")
    ours_experiment_chessboard(bit, dir_path, 18000, 6, model_name)