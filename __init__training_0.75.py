"""
@Function:
@Author : ZhangPeiCheng
@Time : 2023/12/12 7:49
"""
import numpy as np
import tensorflow as tf

from model.predict.predict_v1.mae_predictor import mae_predictor, mae_with_spatial_info
from model.predict.predict_v2_conv.mae_predictor import mae_with_conv_block
from model.train.train_v2_conv.train_v2_single_gpu import train_model_with_spatial_info
from test_model.test_effect import contrast_experiment, ours_experiment_chessboard
from train_set_build.train_change_rate import get_training_change_rate
from train_set_build.train_conv_set import get_training_set

pic_cnt = 19900
# 这个地方排除掉最后一些图片来进行验证
block_cnt = 200
block_sizes = [8]
directory_name = r"D:\czp\combineData"
epoch = 1000
batch_size = 800
test_file_path = "./test_data"
initial_learning_rate = 5e-4


def generate_index(input):
    """generate mask/unmask index"""
    cnt = 0
    mask_index = []
    unmask_index = []
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            if input[i, j] == -1:
                mask_index.append(cnt)
            else:
                unmask_index.append(cnt)
            cnt += 1
    return np.array(mask_index), np.array(unmask_index)


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
    for block_size in block_sizes:
        print("============================= block size: " + str(block_size) + "====================================")
        encoder_input, decoder_output, spatial_mask_infos = get_training_change_rate(directory_name,
                                                                                                 pic_cnt, block_cnt,
                                                                                                 block_size=block_size)
        # encoder_input一定是掩码之后的块
        mask_index, unmask_index = generate_index(encoder_input[0])
        unmask_index = tf.convert_to_tensor(unmask_index)
        mask_index = tf.convert_to_tensor(mask_index)
        encoder_input = encoder_input.reshape((encoder_input.shape[0], encoder_input.shape[1]
                                               , encoder_input.shape[2], 1))

        decoder_output = np.reshape(decoder_output, (decoder_output.shape[0], decoder_output.shape[1], 1))
        model = mae_with_conv_block(encoder_input.shape[1:], spatial_mask_infos.shape[1:], mask_index, unmask_index,
                                    filter=16)
        # training ......
        hist = train_model_with_spatial_info(epoch, batch_size, encoder_input, spatial_mask_infos, decoder_output,
                                             model, "mask_rate_0.75_new", initial_learning_rate)
