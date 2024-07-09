"""
@Function:  用于测试模型的效果
@Author : ZhangPeiCheng
@Time : 2023/12/7 8:37
"""
import os
import random
import numpy as np
from keras.saving.save import load_model

from train_set_build.train_set import open_picture, random_block_split, masking_block, get_mask_surround_pixel


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
    return error_cnt / len(decode), error_cnt


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
        x = random.randint(1 + 20, 510 - 20)
        y = random.randint(1 + 20, 510 - 20)
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
        rate2, error_cnt2 = error_judge(de_info_2, info)
        print("tradition experiment:")
        print(str(filename) + ":" + ": 错误率为：" + str(rate2))
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
    for i in range(len(image_blocks)):
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


def predict_process(encoder_inputs, spatial_mask_infos, model_name):
    """加载模型并返回结果"""
    encoder_inputs = np.reshape(encoder_inputs, (encoder_inputs.shape[0], encoder_inputs.shape[1], 1))
    # mask_vector = np.full(encoder_inputs.shape,fill_value=-1)
    model = load_model(model_name)
    predict_res = model.predict([encoder_inputs, spatial_mask_infos])
    return predict_res


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
        origin_pixel = origin[i]
        # print("预测像素:" + str(predict_pixel) + "," + "加密像素:" + str(encrypt_pixel) + "解密像素:" + str(
        #     decrypt_pixel) + "原始像素:" + str(origin_pixel))
        res1 = np.abs(predict_pixel - encrypt_pixel)
        res2 = np.abs(predict_pixel - decrypt_pixel)
        if res1 < res2:
            # 如果翻转之后残差增大，则原来是原像素，后来是翻转像素
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
        # print(decrypt_result_block)
        decrypt_result.extend(decrypt_result_block)
    return np.array(decrypt_result)


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
        info = generate_binary_stream_for_block(len(indexes), block_cnt)
        # 每个块只有一半的空间可以嵌入信息，暂定第一个像素是嵌入信息的像素
        blocks, block_indexes = random_block_split(pic, block_size, block_cnt)
        # 得到嵌入信息的像素序列以及对于嵌入信息的block
        encrypt_pixels, encrypt_blocks, origin = encrypt_chessboard_for_test(blocks, bit, info)
        # 此处只需要使用encoder_inputs和spatial_mask_infos用于模型的输入
        encoder_inputs, decoder_outputs, spatial_mask_infos = masking_block(encrypt_blocks)
        # 一个图像所有选中图像块的预测结果
        predict_result = predict_process(encoder_inputs, spatial_mask_infos, model_name)
        # 对预测结果进行reshape
        predict_result = predict_result.reshape(
            (predict_result.shape[0], predict_result.shape[1] * predict_result.shape[2]))
        decrypt_result = get_residual_result(predict_result, encrypt_pixels, origin, bit, info)
        # 对比嵌入像素和原像素之间的区别
        error_rate, error_cnt = error_judge(decrypt_result, info)
        print("图像:" + str(filename) + "模型错误率:" + str(error_rate) + "," + "错误像素数量: " + str(error_cnt))


if __name__ == "__main__":
    bit = 3
    dir_path = "../test_data_2"
    model_name = "../trained_model/test_model_3"
    contrast_experiment(bit, dir_path, 6000)
    print("=========================[seg]===============================")
    ours_experiment_chessboard(bit, dir_path, 6000, 6, model_name)
