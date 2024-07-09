"""
@Function:  用于样本的分拣，使得样本复杂度平衡
@Author : ZhangPeiCheng
@Time : 2023/12/12 9:46
"""

import os
import random
import numpy as np
from PIL import Image

from train_set_build.train_conv_set import random_block_split, get_mask_surround_pixel, open_picture, masking_block

def type_judge_blocks(SL, cnt):
    """
    calc the SL type
    """
    if SL >= 40 * cnt:
        return 1
    elif 20 * cnt <= SL < 40 * cnt:
        return 2
    elif 20 * cnt > SL >= 8 * cnt:
        return 3
    elif 8 * cnt > SL >= 0:
        return 4
    elif -8 * cnt <= SL < 0:
        return 5
    elif -20 * cnt <= SL < -8 * cnt:
        return 6
    else:
        return 6


def get_SL(block, index):
    """
    get SL result
    """
    block = block.astype(np.int32)
    i, j = index
    n, m = block.shape
    # 4 corners (i == n-1 and j == 0) or (i == 0 and j == m-1) or (i == n-1 and j == m-1)
    if i == 0 and j == 0:
        SL = np.abs(block[index[0]+1, index[1]] - block[index[0], index[1] + 1])
        return SL
    if i == 0 and j == m - 1:
        SL = np.abs(block[index[0], index[1]-1] - block[index[0]+1, index[1]])
        return SL
    if i == n - 1 and j == 0:
        SL = np.abs(block[index[0]-1, index[1]] -
                    block[index[0], index[1]+1])
        return SL
    if i == n - 1 and j == m - 1:
        SL = np.abs(block[index[0]-1, index[1]] -
                    block[index[0], index[1]-1])
        return SL
    # up or bottom
    if i == 0 or i == n - 1:
        SL = np.abs(block[index[0], index[1]-1] - block[index[0], index[1]+1])
        return SL
    # left or right
    if j == 0 or j == m - 1:
        SL = np.abs(block[index[0]-1, index[1]] - block[index[0]+1, index[1]])
        return SL
    # center
    SL = np.abs(block[index[0], index[1] - 1] - block[index[0], index[1] + 1]) - np.abs(
        block[index[0] - 1, index[1]] - block[index[0] + 1, index[1]])
    return SL
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

def get_complex(block, index):
    block = block.astype(np.int32)
    i,j = index
    # 4 corners (i == n-1 and j == 0) or (i == 0 and j == m-1) or (i == n-1 and j == m-1)
    pixel_list = [block[i, j - 1],
                  block[i, j + 1],
                  block[i + 1, j],
                  block[i - 1, j],
                  ]
    temp = pixel_list
    complex = 0
    for i in range(len(pixel_list)):
        for j in range(3):
            if i != j:
                complex += np.abs(pixel_list[i] - temp[j])
    return complex

def get_sample_len(scala_1,scala_2,scala_3,scala_4,scala_5,scala_6):
    """
    calc the total_cnt
    """
    print(len(scala_1)+len(scala_2)+len(scala_3)+len(scala_4)+len(scala_5)+len(scala_6))
    return len(scala_1)+len(scala_2)+len(scala_3)+len(scala_4)+len(scala_5)+len(scala_6)


def blocks_judge(block):
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
    return np.sum(Sl_for_block), cnt


def block_generator(directory, block_cnt, block_size, random_cnt):
    """
    the body of train set building
    change the pick strategy to random pick
    """
    seed_value = random.randint(0,9999)
    random.seed(seed_value)
    blocks_all = []
    for i in range(random_cnt):
        filename = str(random.randint(1,9985))+".pgm"
        filepath = os.path.join(directory, filename)
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        blocks, block_indexes = random_block_split(pic, block_size, block_cnt)
        blocks_all.extend(blocks)
    return np.array(blocks_all)


scala_1 = []
scala_2 = []
scala_3 = []
scala_4 = []
scala_5 = []
scala_6 = []

def save_dataset(scala_blocks):
    """
    because the high cost of building this dataset, just trying to save all blocks into a csv file
    :param scala_blocks:
    :return:
    """




def sample_sorter(sample_cnt,directory):
    """
    using sample_sorter to balance the number of different complex scala of block
    @implement:
    using the so-called "block" SL to judge the complex scala
    "block" SL: the sum of the SL for every single "cross" set, and the judge standard just blocks * SL standard
    @attention: the aim of the function is to balance scala for only one image!
    """
    # should promise the len of follow 8 array have same scalc
    # this is to regulate that one block must provide different kind of scala set that is not proper
    # correct process should be extract all block from all pic and try to fill 8 arrays below
    all_blocks = block_generator(directory, block_cnt=10, block_size=6, random_cnt=10000)
    scala = []
    while True:
        for i in range(len(all_blocks)):
            # 如果六组的array全部都慢掉了，那么就直接break就好了
            block = all_blocks[i]
            Sl_for_block, units_cnt = blocks_judge(block)
            # 将SL_sum > 220的点给抽取出来
            if Sl_for_block > 220:
                scala.append(block)
            scala.append(block)
            if len(scala) == sample_cnt:
                break
        print("每轮块找到的scala block"+str(len(scala)))
        if len(scala) == sample_cnt:
            break
        all_blocks = block_generator(directory, block_cnt=10, block_size=6,random_cnt=10000)
    return np.array(scala)

def get_training_set_scala(directory, sample_cnt, block_size):
    """
    the body of train set building
    """
    decoder_outputs_all = []
    spatial_mask_infos_all = []
    blocks_all = []
    blocks = sample_sorter(sample_cnt=sample_cnt,directory=directory)
    encoder_inputs, decoder_outputs, spatial_mask_infos, chessboard_blocks = masking_block(blocks)
    decoder_outputs_all.extend(decoder_outputs)
    spatial_mask_infos_all.extend(spatial_mask_infos)
    blocks_all.extend(chessboard_blocks)
    return np.array(blocks_all), np.array(decoder_outputs_all), np.array(spatial_mask_infos_all)


directory_name = r"D:\czp\RIDHproject\BOSSbase_1.01"
if __name__ == "__main__":
    encoder_input, decoder_output,spatial_mask_infos_all = get_training_set_scala(directory_name, 60, 6)
    print(encoder_input.shape)
    print(decoder_output.shape)
    print(spatial_mask_infos_all[0:10])






