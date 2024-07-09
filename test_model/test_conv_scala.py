"""
@Function:             
@Author : ZhangPeiCheng
@Time : 2023/12/19 10:27
"""
import os
import random
import numpy as np
from keras.saving.save import load_model

from train_set_build.sample_sorter import get_SL
from train_set_build.train_conv_set import open_picture, random_block_split, masking_block, get_mask_surround_pixel


def Chen_predict(img, index):
    """
    chen predict method
    :param img:
    :param index:
    :return:
    """
    img = img.astype(np.int32)
    # t1 t3 t2 t4 下，右，上，左
    up = img[index[0] - 1, index[1]]
    dp = img[index[0] + 1, index[1]]
    lp = img[index[0], index[1] - 1]
    rp = img[index[0], index[1] + 1]
    if up == dp == lp == rp:
        # 如果四个值都相等，那么W90就是1
        predict_p = (up + dp) / 2
        return predict_p, np.abs(predict_p - img[index[0], index[1]])
    S90 = np.array([up, dp, (up + dp) / 2])
    S180 = np.array([lp, rp, (lp + rp) / 2])
    u = (up + dp + lp + rp) / 4
    C90 = (np.sum((S90 - u) ** 2)) / 3
    C180 = (np.sum((S180 - u) ** 2)) / 3
    W90 = C180 / (C90 + C180)
    predict_p = W90 * (up + dp) / 2 + (1 - W90) * (lp + rp) / 2
    return predict_p, np.abs(predict_p - img[index[0], index[1]])


def generate_binary_stream(cnt):
    # cnt是要生成的0和1的个数
    # 返回一个由0和1组成的字符串
    stream = []
    for i in range(cnt):
        stream.append(random.randint(0, 1))  # 在流中追加一个随机的0或1
    return stream


def generate_binary_stream_for_block(cnt, block_cnt):
    # cnt是要生成的0和1的个数
    # 返回一个由0和1组成的字符串
    stream = []
    seq_cnt = int(cnt / block_cnt)  # 每一块需要承载的信息数量
    for i in range(int(block_cnt)):
        temp = []
        for j in range(seq_cnt):
            temp.append(random.randint(0, 1))  # 在流中追加一个随机的0或1
        stream.append(temp)
    return stream


def encryptor(pic, bit, indexes, embedding_info=None):
    bit = 2 ** (bit - 1)
    em_pic = np.zeros((512, 512), dtype=np.int32)
    en = []  # 返回加密的像素
    for i in range(512):
        for j in range(512):
            em_pic[i, j] = pic[i, j]
    for i in range(len(indexes)):
        if embedding_info[i] == 1:
            em_pic[indexes[i, 0], indexes[i, 1]] = em_pic[indexes[i, 0], indexes[i, 1]] ^ bit
        en.append(em_pic[indexes[i, 0], indexes[i, 1]])
    return em_pic, np.array(en)


def calc_res(en_pixel_user, bit, pic_1, pic_2, pic, indexes):
    bit = 2 ** (bit - 1)
    decode_1 = []  # 记录解密的信息
    decode_2 = []  # 记录解密的信息
    for i in range(len(indexes)):
        # 残差计算
        predict_p, res3 = Chen_predict(pic_1, indexes[i])
        predict_p, res4 = Chen_predict(pic_2, indexes[i])
        if res3 <= res4:
            decode_2.append(0)
        else:
            decode_2.append(1)
    return np.array(decode_2)


def error_judge(decode, info):
    info = np.array(info)
    info = info.flatten()
    error_cnt = 0
    for i in range(len(decode)):
        if decode[i] != info[i]:
            error_cnt += 1
    return error_cnt


def tradition(indexes, pic):
    res_chen = []
    for i in range(len(indexes)):
        res_chen.append(Chen_predict(img=pic, index=indexes[i]))
    return np.array(res_chen)


def encryptor_user(pic, bit, indexes, info):
    bit = 2 ** (bit - 1)
    em_pic = np.zeros((512, 512), dtype=np.int32)
    en = []  # 返回加密的像素
    for i in range(512):
        for j in range(512):
            em_pic[i, j] = pic[i, j]
    for i in range(len(indexes)):
        em_pic[indexes[i, 0], indexes[i, 1]] = em_pic[indexes[i, 0], indexes[i, 1]] ^ bit
        en.append(em_pic[indexes[i, 0], indexes[i, 1]])
    return em_pic, np.array(en)


def random_pixel_index(pixel_cnt):
    """
    generate random index
    :param pixel_cnt:
    :return:
    """
    indexes = []
    choiced = []
    for i in range(pixel_cnt):
        x = random.randint(1, 510)
        y = random.randint(1, 510)
        while [x, y] in choiced:
            x = random.randint(1, 510)
            y = random.randint(1, 510)
        indexes.append([x, y])
        choiced.append([x, y])
    return np.array(indexes)


def contrast_experiment(bit, dir_path, pixel_cnt):
    """
    传统方法的预测
    :param dir_path:
    :param bit:
    :param pixel_cnt:
    :return:
    """
    for filename in os.listdir(dir_path):
        filepath = os.path.join(dir_path, filename)
        # 打开图片
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        # 随机选择像素点索引加密
        indexes = random_pixel_index(pixel_cnt)
        # 生成加密信息
        info = generate_binary_stream(len(indexes))
        # 首次异或:加密
        en_pic, en_pixel = encryptor(pic, bit, indexes, info)
        temp = en_pic
        # 对于加密图像用户再次进行异或：解密
        en_pic_user, en_pixel_user = encryptor_user(en_pic, bit, indexes, info)
        de_info_2 = calc_res(en_pixel_user, bit, temp, en_pic_user, pic, indexes)
        error_cnt2 = error_judge(de_info_2, info)
        print("tradition experiment:")
        print(str(filename) + ":" + ": 错误率为：" + str(error_cnt2 / len(info)))
        print(str(filename) + ":" + ": 错误数量为：" + str(error_cnt2))


def encrypt_chessboard_for_test(image_blocks, bit, info_for_block):
    """
    第一个是预留像素，第二个是加密像素，这样的顺序进行
    嵌入规则：如果info[j] == 1 则翻转bit位，0则不进行翻转

    :param image_block:
    :return: 嵌入信息的像素、嵌入信息后图像块、原始信息
    """
    bit = 2 ** (bit - 1)
    encrypt_pixels = []
    chessboard_blocks = []
    origins = []
    for i in range(len(info_for_block)):
        image_block = image_blocks[i]
        info = info_for_block[i]
        m, n = image_block.shape
        encrypt_pixel = []
        origin = []
        chessboard_block = np.full(image_block.shape, fill_value=-1)
        cnt = 0
        for i in range(m):
            is_odd_row = (i + 1) % 2 == 1
            for j in range(n):
                is_odd_column = (j + 1) % 2 == 1
                if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                    # print("嵌入信息:" + str(info[j]))
                    # 如果行号同奇同偶，则嵌入信息
                    origin.append(image_block[i, j])
                    if info[cnt] == 1:
                        # 如果隐藏信息为1，则进行异或，否则不进行异或
                        encrypt_pixel.append(image_block[i, j] ^ bit)
                        chessboard_block[i, j] = image_block[i, j] ^ bit
                    else:
                        encrypt_pixel.append(image_block[i, j])
                        chessboard_block[i, j] = image_block[i, j]
                    cnt += 1
                else:
                    # 否则保留像素
                    chessboard_block[i, j] = image_block[i, j]
        # print("嵌入信息数量"+str(cnt))
        encrypt_pixels.append(encrypt_pixel)
        chessboard_blocks.append(chessboard_block)
        origins.append(origin)
    return np.array(encrypt_pixels), np.array(chessboard_blocks), np.array(origins)


def predict_process(chessboard_blocks_simple, chessboard_blocks_complex, spatial_mask_infos_simple,
                    spatial_mask_infos_complex
                    , model_name_simple, model_name_complex):
    """加载模型并返回结果"""
    chessboard_blocks_simple = np.reshape(chessboard_blocks_simple, (
        chessboard_blocks_simple.shape[0], chessboard_blocks_simple.shape[1], chessboard_blocks_simple.shape[2], 1))
    chessboard_blocks_complex = np.reshape(chessboard_blocks_complex, (
        chessboard_blocks_complex.shape[0], chessboard_blocks_complex.shape[1], chessboard_blocks_complex.shape[2], 1))
    model_simple = load_model(model_name_simple)
    model_complex = load_model(model_name_complex)
    predict_res_simple = model_simple.predict([chessboard_blocks_simple, spatial_mask_infos_simple])
    predict_res_complex = model_complex.predict([chessboard_blocks_complex, spatial_mask_infos_complex])
    return predict_res_simple, predict_res_complex


def decrypt_for_block(encrypt_blocks, bit):
    """
    将block的所有嵌入信息的位置进行bit位的异或
    """
    user_decrypt_pixels = []
    for i in range(len(encrypt_blocks)):
        encrypt_block = encrypt_blocks[i]
        m, n = encrypt_block.shape
        user_decrypt_pixel = []
        for i in range(m):
            is_odd_row = (i + 1) % 2 == 1
            for j in range(n):
                is_odd_column = (j + 1) % 2 == 1
                if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                    # 有嵌入信息的位置的像素全部进行再次异或
                    user_decrypt_pixel.append(encrypt_block[i, j] ^ bit)
        user_decrypt_pixels.append(user_decrypt_pixel)
    return np.array(user_decrypt_pixels)


def blocks_avg_sl(block):
    """
    calc the total SL value for a block
    """
    m, n = block.shape
    Sl_for_block = []
    cnt = 0  # the number of calculate units
    for i in range(m):
        is_odd_row = (i + 1) % 2 == 1
        for j in range(n):
            is_odd_column = (j + 1) % 2 == 1
            if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                SL = get_SL(block, (i, j))  # single SL value for one pixel
                Sl_for_block.append(SL)
                cnt += 1
    return np.sum(Sl_for_block)


def random_block_split(image, block_size, cnt):
    """
    Random selection of image blocks
    """
    block_indexes_simple = []
    block_indexes_complex = []
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    # Row index of a block and column index of a block
    num_blocks_row = height // block_size
    num_blocks_col = width // block_size
    selected_blocks_simple = []
    selected_blocks_complex = []
    selected_indices = np.random.choice(num_blocks_row * num_blocks_col, cnt, replace=False)
    total_len = len(selected_indices)
    def get_block(total_cnt,image_array):
        selected_indices = np.random.choice(num_blocks_row * num_blocks_col, cnt, replace=False)
        blocks = []
        for i in range(total_cnt):
            idx = selected_indices[i]
            row = idx // num_blocks_col
            col = idx % num_blocks_col
            start_row, start_col = row, col
            if start_row + block_size > height or start_col + block_size > width:
                i -= 1
                continue
            block = image_array[start_row: start_row + block_size, start_col:start_col + block_size]
            blocks.append(block)
        return np.array(blocks)

    blocks = get_block(total_cnt=len(selected_indices), image_array=image_array)
    cnt = 0
    while True:
        try:
            block = blocks[cnt]
        except:
            blocks = get_block(total_cnt=len(selected_indices), image_array=image_array)
            cnt = 0
            block = blocks[cnt]
        # 这个地方需要进行判断块的scala
        SL = blocks_avg_sl(block)
        if SL < 220 and len(selected_blocks_simple) < int(total_len/2):
            selected_blocks_simple.append(block)
        elif SL >= 220 and len(selected_blocks_complex) < int(total_len/2):
            selected_blocks_complex.append(block)
        cnt += 1
        if len(selected_blocks_simple) == int(total_len/2) and len(selected_blocks_complex) == int(total_len/2):
            break
    return selected_blocks_simple, selected_blocks_complex


def block_res_calc(predict_result, encrypt_pixels, decrypt_pixels, origin, bit, info_block):
    """
    残差的计算
    :param predict_result: 模型预测结果
    :param encrypt_pixels: 加密的像素
    :param decrypt_pixels: 加密像素的异或结果
    :return:
    """
    decrypt_result = []
    for i in range(len(predict_result)):
        encrypt_pixel = encrypt_pixels[i]
        decrypt_pixel = encrypt_pixels[i] ^ bit
        predict_pixel = predict_result[i]
        res1 = np.abs(predict_pixel - encrypt_pixel)
        res2 = np.abs(predict_pixel - decrypt_pixel)
        if res1 < res2:
            decrypt_result.append(0)
        else:
            decrypt_result.append(1)
    return np.array(decrypt_result)


def get_residual_result(predict_result, encrypt_pixels, origin, bit, info):
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
        decrypt_result.extend(decrypt_result_block)
    return np.array(decrypt_result)


def ours_experiment_chessboard(bit, dir_path, pixel_cnt, block_size, model_name_simple, model_name_complex):
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
        pixel_cnt = int(pixel_cnt / 2)
        block_cnt = int(pixel_cnt / (block_size * block_size / 2))
        if pixel_cnt % block_size != 0:
            print("需要pixel_cnt可以被block_size整除")
            return
        # 随机选择像素点索引加密
        indexes = random_pixel_index(pixel_cnt)
        # 生成加密信息, 将加密信息只进行简单的切分之后[[block_info1...], [block_info2...], ...]
        # 每种不同的模型
        info_simple = generate_binary_stream_for_block(int(len(indexes)), int(block_cnt))
        info_complex = generate_binary_stream_for_block(int(len(indexes)), int(block_cnt))
        # 每个块只有一半的空间可以嵌入信息，暂定第一个像素是嵌入信息的像素
        # 此处只需要使用encoder_inputs和spatial_mask_infos用于模型的输入
        selected_blocks_simple, selected_blocks_complex = random_block_split(
            pic, block_size, block_cnt * 2)  # 将块的数量*2
        # 分别选择复杂块与平滑块
        encrypt_pixels_simple, encrypt_blocks_simple, origin_simple = (
            encrypt_chessboard_for_test(selected_blocks_simple, bit, info_simple))
        encrypt_pixels_complex, encrypt_blocks_complex, origin_complex = (
            encrypt_chessboard_for_test(selected_blocks_complex, bit, info_complex))

        encoder_inputs_simple, decoder_outputs_simple, spatial_mask_infos_simple, chessboard_blocks_simple = masking_block(
            selected_blocks_simple)
        encoder_inputs_complex, decoder_outputs_complex, spatial_mask_infos_complex, chessboard_blocks_complex = masking_block(
            selected_blocks_complex)
        # print("网络输入")
        # print(encoder_inputs_simple.shape)
        # print(encoder_inputs_complex.shape)
        # 一个图像所有选中图像块的预测结果
        predict_result_simple, predict_result_complex = predict_process(chessboard_blocks_simple,
                                                                        chessboard_blocks_complex,
                                                                        spatial_mask_infos_simple,
                                                                        spatial_mask_infos_complex,
                                                                        model_name_simple, model_name_complex)
        # 对预测结果进行reshape
        predict_result_simple = predict_result_simple.reshape(
            (predict_result_simple.shape[0], predict_result_simple.shape[1] * predict_result_simple.shape[2]))
        predict_result_complex = predict_result_complex.reshape(
            (predict_result_complex.shape[0], predict_result_complex.shape[1] * predict_result_complex.shape[2]))
        # print("网络预测结果")
        # print(predict_result_simple.shape)
        # print(predict_result_complex.shape)
        decrypt_result_simple = get_residual_result(predict_result_simple, encrypt_pixels_simple, origin_simple, bit,
                                                    info_simple)
        decrypt_result_complex = get_residual_result(predict_result_complex, encrypt_pixels_complex, origin_complex,
                                                     bit, info_complex)
        # 对比嵌入像素和原像素之间的区别
        # print("网络解码结果")
        # print(decrypt_result_simple.shape)
        # print(decrypt_result_complex.shape)
        error_cnt_simple = error_judge(decrypt_result_simple, info_simple)
        error_cnt_complex = error_judge(decrypt_result_complex, info_complex)
        error_rate = (error_cnt_simple + error_cnt_complex) / (pixel_cnt*2)
        print("图像:" + str(filename) + "模型错误率:" + str(error_rate) + "," + "错误像素数量: " + str(
            error_cnt_simple + error_cnt_complex))
        pixel_cnt = pixel_cnt * 2


import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    bit = 3
    dir_path = "../test_data"
    model_name_simple = "../trained_model/test_model_conv_4"
    model_name_complex = "../trained_model/complex_model_conv_1"
    contrast_experiment(bit, dir_path, 12000)
    print("=========================[seg]===============================")
    ours_experiment_chessboard(bit, dir_path, 12000, 6, model_name_simple, model_name_complex)
