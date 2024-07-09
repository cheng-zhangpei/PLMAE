"""
@Function:             
@Author : ZhangPeiCheng
@Time : 2024/2/21 19:20
"""
import  tensorflow as tf

from test_model.test_random_rate import ours_experiment_chessboard

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    # 指定使用第一块GPU来进行训练
    tf.config.experimental.set_visible_devices(gpus[3], 'GPU')
    bit = 3
    dir_path = "../test_data"
    model_name = "../trained_model/index_change_8_25%"
    # model_name = "../trained_model/test_model_combine_1"
    print("=========================[seg]===============================")
    ours_experiment_chessboard(bit, dir_path, 6000, 8, model_name,masking_rate=0.75)