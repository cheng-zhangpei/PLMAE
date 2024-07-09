"""
@Function: 修改random mask rate 测试接口
@Author : ZhangPeiCheng
@Time : 2024/2/15 15:24
"""
import numpy as np
from keras.saving.save import load_model
from train_set_build.train_random_mask import masking_block

from train_set_build.train_change_rate import open_picture

def predict_process(chessboard_blocks, spatial_mask_infos, model_name):
    """加载模型并返回结果
        独立出来不会导致显存冲突
    """
    chessboard_blocks = np.reshape(chessboard_blocks, (chessboard_blocks.shape[0], chessboard_blocks.shape[1],chessboard_blocks.shape[2], 1))
    model = load_model(model_name)
    predict_res = model.predict([chessboard_blocks, spatial_mask_infos])
    return predict_res

def predict_by_model_2(block_path,model_path,masking_rate):
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
    print("原始块")
    print(blocks[0])
    encoder_inputs, decoder_outputs, spatial_mask_infos, chessboard_blocks, origin = masking_block(blocks, masking_rate)
    # 预测
    print("放入网络的像素")
    print(chessboard_blocks[0])
    predict_value = predict_process(chessboard_blocks, spatial_mask_infos,model_path)

    return predict_value.flatten()
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

if __name__ == '__main__':
    # 测试目录中block size=8*8
    masking_rate = 0.75
    print(gpus)
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    # blocks_directory = r"D:\czp\mae\test_model\error_block_10"
    block_path = r"D:\czp\mae\test_model\error_block_10\img-airplane.bmp-pos-(0, 8).bmp"
    test_block = open_picture(block_path)
    test_block = test_block.astype(np.int32)
    # 如果是改block=6 / 10 可以把结尾8改成对应size
    model_path_75 = r"D:\czp\mae\trained_model\index_change_8_random" # 掩码0.75模型的
    model_path_50 = r"D:\czp\mae\trained_model\index_change_8_random_50%" # 掩码0.5
    # 注：masking_rate需要随着模型进行变化，否则输出的预测值无法对应
    predict_values_2 = predict_by_model_2(block_path,model_path_75,masking_rate=masking_rate)
    print("预测结果")
    print(predict_values_2)
