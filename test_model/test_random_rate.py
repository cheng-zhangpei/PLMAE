"""
@Function:  不同掩码率的测试
@Author : ZhangPeiCheng
@Time : 2024/2/14 14:10
"""
import os
import random
import numpy as np
from PIL import Image
from keras.saving.save import load_model

from test_model.test_conv import random_pixel_index, generate_binary_stream_for_block, encrypt_chessboard_for_test, \
    predict_process, get_residual_result, block_res_calc, error_judge
from train_set_build.train_conv_set import open_picture, random_block_split, get_training_set
from train_set_build.train_random_mask import masking_block


def fill_image_block(block):
    """
    进行图像填充
    :param block:
    :return:
    """
    block[0] = block[2]  # 第0行用第二行进行填充
    block[block.shape[0] - 1] = block[block.shape[0] - 3]  # 第0行用第二行进行填充
    block[:, block.shape[1] - 1] = block[:, block.shape[1] - 3]  # 第0行用第二行进行填充
    block[:, 0] = block[:, 2]
    return block


def random_block_split_(image, block_size, cnt):
    """
    Random selection of image blocks
    返回两种类型的块
    """
    block_indexes = []
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    # Row index of a block and column index of a block
    num_blocks_row = height // block_size
    num_blocks_col = width // block_size
    selected_indices = np.random.choice(num_blocks_row * num_blocks_col, cnt, replace=False)
    selected_blocks = []
    selected_blocks_ = []
    for idx in selected_indices:
        row = idx // num_blocks_col
        col = idx % num_blocks_col
        start_row, start_col = row, col
        if start_row + block_size > height or start_col + block_size > width:
            continue
        if start_row == 0 or start_col == 0:
            continue
        try:
            block = image_array[start_row: start_row + block_size, start_col:start_col + block_size]
            block_ = image_array[start_row - 1: start_row + block_size + 1, start_col - 1:start_col + block_size + 1]
        except:
            # 如果越界了我懒得处理了，就直接简单的抛出就好了
            continue
        selected_blocks.append(block)
        selected_blocks_.append(block_)
        if block_.shape != selected_blocks_[0].shape:
            print("出现异常")
            print(block_.shape)
            print(selected_blocks_[0].shape)
            print([start_row, start_col])
        block_indexes.append([start_row, start_col])
    selected_blocks_ = np.array(selected_blocks_)
    selected_blocks = np.array(selected_blocks)
    return selected_blocks, np.array(block_indexes), np.array(selected_blocks_)


def block_split(image, block_size):
    """
    将一个图像块分为block_size的块
    """
    # 获取图像的高度和宽度
    height, width = image.shape

    # 计算每个块的高度和宽度
    block_height = block_size
    block_width = block_size

    blocks = []
    indices = []
    for i in range(0, height, block_height):
        for j in range(0, width, block_width):
            # 切割图像
            block = image[i: i + block_height, j: j + block_width]
            if block.shape != (block_size, block_size):
                continue
            blocks.append(block)
            indices.append([i, j])
    return np.array(blocks), np.array(indices)


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
        return predict_p
    S90 = np.array([up, dp, (up + dp) / 2])
    S180 = np.array([lp, rp, (lp + rp) / 2])
    u = (up + dp + lp + rp) / 4
    C90 = (np.sum((S90 - u) ** 2)) / 3
    C180 = (np.sum((S180 - u) ** 2)) / 3
    W90 = C180 / (C90 + C180)
    predict_p = W90 * (up + dp) / 2 + (1 - W90) * (lp + rp) / 2
    return predict_p


def block_res_calc_(predict_result, encrypt_pixels, bit):
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


def get_residual_result_(predict_result, encrypt_pixels, bit):
    """
    :return:
    """
    bit = 2 ** (bit - 1)
    decrypt_result = []
    # print("解密时候的信息验证")
    # print(encrypt_pixels[0:3])
    for i in range(len(predict_result)):
        # 获得一个block的像素
        encrypt_block_pixel = encrypt_pixels[i]
        predict_block_pixel = predict_result[i]
        decrypt_result_block = block_res_calc_(predict_block_pixel, encrypt_block_pixel, bit)
        decrypt_result.append(decrypt_result_block)
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


def retraining_block_judge_(decrypt_blocks_info, info):
    """
    这个记录的只是单个块的解密信息
    """
    info = np.array(info)
    wrong_index = []  # 记录在解密过程中错误的索引
    all_wrong_cnt = []  # 记录在解密过程中错误的索引
    wrong_cnt = 0
    for j in range(len(decrypt_blocks_info)):
        if decrypt_blocks_info[j] != info[j]:
            wrong_cnt += 1
    all_wrong_cnt.append(wrong_cnt)
    return np.array(all_wrong_cnt), np.array(wrong_index)


def get_contrast_for_block(blocks):
    """
    将传统的方法用在块中
    为了保证预测的准确性，此处的blocks的大小将会比原来的块大一圈

    :return: 对于每一个块的预测信息结果
    """
    predict_for_blocks = []
    tra_ori_pixels = []
    for k in range(len(blocks)):
        block = blocks[k]
        block = np.pad(block, pad_width=1, constant_values=0)
        block = fill_image_block(block)
        m, n = block.shape
        tra_block_predict = []
        for i in range(m):
            if i == 0 or i == m - 1:
                # 第一行和最后一行不能进行信息的嵌入
                continue
            is_odd_row = i % 2 == 1
            for j in range(n):
                if j == 0 or j == n - 1:
                    # 第一列和最后一列不能进行信息的嵌入
                    continue
                is_odd_column = j % 2 == 1
                if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                    # 对信息进行加密
                    index = [i, j]
                    tra_ori_pixels.append(block[i, j])
                    predict_value = Chen_predict(block, index)
                    tra_block_predict.append(predict_value)
        predict_for_blocks.append(tra_block_predict)
    return predict_for_blocks, np.array(tra_ori_pixels)


def calc_res(bit, en_1, predict_vector):
    """

    :param bit:
    :param en_1: 第一次加密
    :param en_2:
    :param predict_vector:
    :return:
    """
    bit = 2 ** (bit - 1)
    predict_values_blocks = []
    # print(len(predict_vector[0]))
    for k in range(len(predict_vector)):
        decode = []  # 记录解密的信息
        en_block = en_1[k]
        predict_vector_ = predict_vector[k]
        for i in range(len(predict_vector[1])):
            # 残差计算
            predict_value = predict_vector_[i]
            res3 = np.abs(en_block[i] - predict_value)  # 第一次加密结果
            res4 = np.abs((en_block[i] ^ bit) - predict_value)  # 第二次加密结果
            # try:
            if res3 < res4:
                decode.append(0)
            else:
                decode.append(1)
        predict_values_blocks.append(decode)
    return np.array(predict_values_blocks)


def encrypt_chessboard_for_test_tra(image_blocks, bit, info_for_block):
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
        encrypt_pixels.append(encrypt_pixel)
        chessboard_blocks.append(chessboard_block)
        origins.append(origin)
    return np.array(encrypt_pixels), np.array(chessboard_blocks), np.array(origins)


def generate_binary_stream_change(block_size, block_cnt, masking_rate):
    # cnt是要生成的0和1的个数
    # 返回一个由0和1组成的字符串
    stream = []
    seq_cnt = int((block_size * block_size) * masking_rate)  # 每一块需要承载的信息数量
    # print(seq_cnt)
    for i in range(int(block_cnt)):
        temp = []
        for j in range(seq_cnt):
            temp.append(random.randint(0, 1))  # 在流中追加一个随机的0或1
        stream.append(temp)
    return stream


def save_block(block, filename):
    """
    将符合要求的块进行保存
    :param block:
    :return:
    """
    block = block.astype(np.uint8)
    block = Image.fromarray(block)
    save_directory = "./error_block_10/"
    block.save(save_directory + filename + ".bmp")


def select_block(info, decrypt_result, decrypt_result_tra, ori_block, block_indexes, filename):
    """
    将网络效果比tradition的错误数的给挑选出来并且将错误数量持久化并且编号
    :param info:
    :param decrypt_result:
    :param decrypt_result_tra:
    :return:
    """
    cnt = 0
    for i in range(len(decrypt_result)):
        row = block_indexes[i][0]
        col = block_indexes[i][1]
        wrong_net, wrong_index = retraining_block_judge_(decrypt_result[i], info[i])
        wrong_tra, wrong_index = retraining_block_judge_(decrypt_result_tra[i], info[i])
        save_block(ori_block[i],
                   "img-" + str(filename) + "-pos-" + str((row, col)) + "-net-" + str(wrong_net) + "-tra-" + str(
                       wrong_tra))
        cnt += 1


def select_block_(ori_block, block_indexes, filename):
    """
    将网络效果比tradition的错误数的给挑选出来并且将错误数量持久化并且编号
    :param info:
    :param decrypt_result:
    :param decrypt_result_tra:
    :return:
    """
    cnt = 0
    for i in range(len(ori_block)):
        row = block_indexes[i][0]
        col = block_indexes[i][1]
        save_block(ori_block[i], "img-" + str(filename) + "-pos-" + str((row, col)))
        cnt += 1


def random_block_split_for_all(blocks, block_cnt, index):
    selected_indices = [random.randint(0, len(blocks) - 1) for _ in range(block_cnt)]
    random_block = []
    random_index = []
    for i in range(len(selected_indices)):
        random_block.append(blocks[selected_indices[i]])
        # print("所选索引")
        # print(index[i])
        random_index.append(index[selected_indices[i]])
    return np.array(random_block), np.array(random_index)


def get_distance(ori, tra, model):
    print("tra shape")
    """获取各个值得绝对误差绝对误差"""
    tra = np.array(tra)
    model = model.flatten()
    tra = tra.flatten()
    return np.abs(np.round(ori - model, 1)), np.abs(np.round(ori - tra, 1))


def analysis(tra_dis, model_dis):
    model_cnt = 0
    tra_cnt = 0
    print(tra_dis.shape)
    model_dis = model_dis.flatten()
    tra_dis = tra_dis.flatten()
    for i in range(len(tra_dis)):
        if tra_dis[i] > 4:
            tra_cnt += 1
        if model_dis[i] > 4:
            model_cnt += 1
    return tra_cnt, model_cnt


def get_model_test_error(model_name):
    directory_name = r"D:\czp\mae\test_data_2"
    encoder_input, decoder_output, spatial_mask_infos_all = get_training_set(directory_name, 15, 10, 6)
    model = load_model(model_name)
    test_res = model.evaluate([encoder_input, spatial_mask_infos_all], decoder_output)
    return test_res[1]


def encrypt_chessboard_change(image_blocks, chessboard_blocks, bit, info_for_block):
    """
    第一个是预留像素，第二个是加密像素，这样的顺序进行
    嵌入规则：如果info[j] == 1 则翻转bit位，0则不进行翻转

    :param image_block:
    :return: 嵌入信息的像素、嵌入信息后图像块、原始信息
    """
    bit = 2 ** (bit - 1)
    encrypt_pixels = []
    encrypted_blocks = []
    origins = []
    for k in range(len(image_blocks)):
        # 拿到图像块和信息
        image_block = image_blocks[k]
        info = info_for_block[k]
        m, n = image_block.shape
        chessboard_block = chessboard_blocks[k]
        encrypt_pixel = []
        origin = []
        encrypted_block = np.full(image_block.shape, fill_value=-1)
        cnt = 0
        for i in range(m):
            for j in range(n):
                if chessboard_block[i, j] != -1:
                    # 是保留像素的话
                    encrypted_block[i, j] = image_block[i, j]
                else:
                    # 否则保留像素，只要调整保留像素的数量就好了
                    # 从块的第一个开始嵌入信息
                    origin.append(image_block[i, j])
                    # 如果隐藏信息为1，则进行异或，否则不进行异或
                    if info[cnt] == 1:
                        # 将嵌入信息的像素放到图像块中去
                        encrypted_block[i, j] = image_block[i, j] ^ bit
                        encrypt_pixel.append(encrypted_block[i, j])
                    else:
                        encrypted_block[i, j] = image_block[i, j]
                        encrypt_pixel.append(encrypted_block[i, j])
                    cnt += 1
        encrypt_pixels.append(encrypt_pixel)
        encrypted_blocks.append(encrypted_block)
        origins.append(origin)
    return np.array(encrypt_pixels), np.array(chessboard_blocks), np.array(origins)

def recover_image(predict_value, chessboards,save_path_predict,save_path_masked):
    predict_blocks = []
    masked_blocks = []
    predicted_image = np.zeros((512, 512))
    masked_image = np.zeros((512, 512))

    for k in range(len(chessboards)):
        chessboard = chessboards[k]
        predict_pixels = predict_value[k].flatten()
        cnt = 0
        predicted_block = np.zeros((chessboard.shape[0], chessboard.shape[1]))
        masked_block = np.zeros((chessboard.shape[0], chessboard.shape[1]))
        for i in range(len(chessboard)):
            for j in range(len(chessboard[0])):
                if chessboard[i,j] == -1:
                    predicted_block[i,j]=np.round(predict_pixels[cnt])
                    masked_block[i,j]=255
                    cnt+=1
                else:
                    predicted_block[i, j]=chessboard[i,j]
                    masked_block[i, j] = 0
        predict_blocks.append(predicted_block)
        masked_blocks.append(masked_block)
    predict_blocks = np.array(predict_blocks)
    block_index = 0
    for y in range(64):
        for x in range(64):
            # 计算当前块的左上角在原始图像中的位置
            start_x = x * 8
            start_y = y * 8
            # 将当前块的像素值填充到原始图像中对应的位置
            predicted_image[start_y:start_y + 8, start_x:start_x + 8] = predict_blocks[block_index]
            masked_image[start_y:start_y + 8, start_x:start_x + 8] = masked_blocks[block_index]
            block_index += 1
    # 将numpy数组转换为PIL图像
    original_image_pil = Image.fromarray(predicted_image.astype('uint8'))
    masked_image_pil = Image.fromarray(masked_image.astype('uint8'))

    # 保存图像
    original_image_pil.save(save_path_predict)
    masked_image_pil.save(save_path_masked)

def ours_experiment_chessboard(bit, dir_path, pixel_cnt, block_size, model_name,masking_rate):
    """
    棋盘格划分确实有一个比较大的问题就是
    :param bit:
    :param dir_path:
    :param pixel_cnt:
    :param block_size:
    :return:
    """
    for filename in os.listdir(dir_path):
        for m in range(1):
            filepath = os.path.join(dir_path, filename)
            # 打开图片
            pic = open_picture(filepath)
            pic = pic.astype(np.int32)
            block_cnt = int(pixel_cnt / (block_size * block_size)*masking_rate)
            if pixel_cnt % block_size != 0:
                print("需要pixel_cnt可以被block_size整除")
                return
            blocks, block_indexes = block_split(pic, block_size)
            # 随机选择块
            # blocks,block_indexes = random_block_split_for_all(blocks,block_cnt,block_indexes)
            info = generate_binary_stream_change(block_size, int(len(blocks)), masking_rate)
            # en_selected_pixel_, en_selected_blocks_, origin = encrypt_chessboard_for_test(selected_blocks_, bit, info)
            # predict_tra, tra_ori_pixels = get_contrast_for_block(en_selected_blocks_)
            decoder_outputs, spatial_mask_infos, chessboard_blocks, origin_block = masking_block(blocks, 0.5)
            # print("放入模型的棋盘格")
            # print(chessboard_blocks[1000])
            # print("原始block")
            # print(origin_block[1000])
            encrypt_pixels, encrypt_blocks, origin = encrypt_chessboard_change(origin_block, chessboard_blocks, bit, info)
            # print("加密像素")
            # print(encrypt_pixels[1000])
            # print("原始块")
            # print(origin_block[1000])
            # # print("label")
            # # print(decoder_outputs[0])
            # print("模型输入")
            # print(chessboard_blocks[1000])
            print(chessboard_blocks.shape)
            predict_result = predict_process(chessboard_blocks, spatial_mask_infos, model_name)
            decrypt_result = get_residual_result_(predict_result, encrypt_pixels, bit)
            # decrypt_result_tra = calc_res(bit, en_selected_pixel_, predict_tra)
            # print("预测值")
            # print(predict_result[1000].flatten())
            # print("加密像素")
            # print(encrypt_pixels[1000])
            # print("加密像素再次异或")
            # print(np.array(encrypt_pixels[1000]) ^ 4)
            # print("原始像素")
            # print(decoder_outputs[1000])
            # print("解密信息")
            # print(decrypt_result[1000].flatten().tolist())
            # print("原始信息")
            # print(info[10])
            filename_without_extension = filename.rsplit(".", 1)[0]

            # save_path_predicted = "../interface/predicted_image/"+str(filename_without_extension)+"/predict/"+str(filename_without_extension)+"-"+str(m)+".bmp"
            # save_path_masked = "../interface/predicted_image/"+str(filename_without_extension)+"/masked/"+str(filename_without_extension)+"-"+str(m)+".bmp"

            save_path_predicted = "./predicted_image/predict/"+str(filename)
            save_path_masked = "./predicted_image/masked/"+str(filename)
            # recover_image(predict_result,chessboard_blocks,save_path_predicted,save_path_masked)
            error_rate, error_cnt = error_judge(decrypt_result.flatten(), info)
            print("图像:" + str(filename) + "模型错误率:" + str(error_rate) + "," + "错误像素数量: " + str(error_cnt))
    # test_error = get_model_test_error(model_name)
    # print("测试集")
    # print(test_error)


import tensorflow as tf

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    bit = 3
    dir_path = "../test_data"
    model_name = "../trained_model/index_change_8_random_50%"
    # model_name = "../trained_model/test_model_combine_1"
    print("=========================[seg]===============================")
    ours_experiment_chessboard(bit, dir_path, 16000,
                               8, model_name, 0.5)
