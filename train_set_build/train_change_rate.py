"""
@Function: 改变掩码率
@Author : ZhangPeiCheng
@Time : 2024/2/13 12:17
"""

import os
import random
import numpy as np
from PIL import Image



def random_block_split(image, block_size, cnt):
    """
    Random selection of image blocks
    """
    block_indexes = []
    image_array = np.array(image)
    height, width = image_array.shape[:2]
    # Row index of a block and column index of a block
    num_blocks_row = height // block_size
    num_blocks_col = width // block_size
    selected_indices = np.random.choice(num_blocks_row * num_blocks_col, cnt, replace=False)
    selected_blocks = []
    for idx in selected_indices:
        row = idx // num_blocks_col
        col = idx % num_blocks_col
        start_row, start_col = row, col
        if start_row + block_size > height or start_col + block_size > width:
            continue
        block = image_array[start_row: start_row + block_size, start_col:start_col + block_size]
        selected_blocks.append(block)
        block_indexes.append([start_row,start_col])
    return selected_blocks,block_indexes


def open_picture(path):
    image = Image.open(path)
    arr = np.asarray(image)
    return arr


def get_surround_change_rate(block, index):
    # seq: [up, down, left, right]
    mask_sur_pixel = []
    x,y = index
    for i in range(x -1, x + 2):
        for j in range(y-1, y+2):
            if i == x and j == y:
                continue
            if i < 0 or j < 0:
                # -1会变成从后往前
                mask_sur_pixel.append(-1)
                continue
            try:
                mask_sur_pixel.append(block[i, j])
            except:
                # 如果出现了越界的情况，就直接用-1来填充
                mask_sur_pixel.append(-1)
    return mask_sur_pixel

def get_surround_change_rate_change(block, index):
    # seq: [up, down, left, right]
    mask_sur_pixel = []
    x,y = index
    for i in range(x-1,x+2):
        is_odd_row = (i + 1) % 2 == 1
        for j in range(y-1, y+2):
            is_odd_column = j % 2 == 1
            if i == x and j == y:
                continue
            if i < 0 or j < 0:
                # -1会变成从后往前
                mask_sur_pixel.append(-1)
                continue
            try:
                if (is_odd_row and is_odd_column) and is_odd_row:
                    # 判断是否是保留像素
                    mask_sur_pixel.append(block[i, j])
                else:
                    mask_sur_pixel.append(-1)
            except:
                # 如果出现了越界的情况，就直接用-1来填充
                mask_sur_pixel.append(-1)
    return mask_sur_pixel

def create_chessboard_blocks(image_block):
    """
    :return: encoder_input   decoder_input(with masked)
    """
    m, n = image_block.shape
    encoder_input = []
    mask_surround_pixels = []
    chessboard_blocks = np.full(image_block.shape, fill_value=-1)
    decoder_output = []
    for i in range(m):
        is_odd_row = (i + 1) % 2 == 1
        for j in range(n):
            is_odd_column = j % 2 == 1
            if (is_odd_row and is_odd_column) and is_odd_row:
                chessboard_blocks[i, j] = image_block[i, j]
                encoder_input.append(image_block[i, j])
            else:
                # 第一个点是加密像素
                decoder_output.append(image_block[i, j])
                # 周围像素集合的选择策略要改变->此处输出的信息一共有八个像素
                # 这个都会放入的像素是还没有进行掩码之后的像素，所以这个最好是要放到外面去
                mask_surround_pixel = get_surround_change_rate_change(image_block, (i, j))
                mask_surround_pixels.append(mask_surround_pixel)
    return np.array(encoder_input), chessboard_blocks, decoder_output,np.array(mask_surround_pixels)


def masking_block(blocks):
    """
    Masking operations on blocks
    Now, let's start by dividing the board into squares
    => Subsequent changes can be made to randomized masks with mask rate adjustments
    """
    chessboard_blocks = []
    spatial_mask_infos = []
    decoder_outputs = []
    origin_block = []
    for block in blocks:
        origin_block.append(block)
        encoder_input, chessboard_block, decoder_output, spatial_mask_info = create_chessboard_blocks(block)
        chessboard_blocks.append(chessboard_block)
        spatial_mask_infos.append(spatial_mask_info)
        decoder_outputs.append(decoder_output)
    return np.array(decoder_outputs), np.array(spatial_mask_infos),np.array(chessboard_blocks)
def random_block_split_for_all(blocks,block_cnt,index):
    selected_indices = [random.randint(0, len(blocks)-1) for _ in range(block_cnt)]
    random_block = []
    random_index = []
    for i in range(len(selected_indices)):
        random_block.append(blocks[selected_indices[i]])
        # print("所选索引")
        # print(index[i])
        random_index.append(index[selected_indices[i]])
    return np.array(random_block), np.array(random_index)
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
def get_training_change_rate(directory, pic_cnt, block_cnt, block_size):
    """
    the body of train set building
    """
    cnt = 0
    decoder_outputs_all = []
    spatial_mask_infos_all = []
    blocks_all = []
    origin_all = []
    for filename in os.listdir(directory):
        if cnt == pic_cnt:
            break
        filepath = os.path.join(directory, filename)
        # open picture
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        # blocks, block_indexes = random_block_split(pic, block_size, block_cnt)
        blocks, block_indexes = block_split(pic, block_size)
        # 随机选择块
        blocks, block_indexes = random_block_split_for_all(blocks, block_cnt, block_indexes)
        decoder_outputs, spatial_mask_infos,chessboard_blocks = masking_block(blocks)


        decoder_outputs_all.extend(decoder_outputs)
        spatial_mask_infos_all.extend(spatial_mask_infos)
        blocks_all.extend(chessboard_blocks)
        cnt += 1
    return np.array(blocks_all), np.array(decoder_outputs_all),np.array(spatial_mask_infos_all)

def get_training_change_mask(directory, pic_cnt):
    """
    the body of train set building
    """
    cnt = 0
    decoder_outputs_all = []
    spatial_mask_infos_all = []
    blocks_all = []
    for filename in os.listdir(directory):
        if cnt == pic_cnt:
            break
        filepath = os.path.join(directory, filename)
        # open picture
        pic = open_picture(filepath)
        pic = pic.astype(np.int32)
        blocks = np.array([pic])
        decoder_outputs, spatial_mask_infos,chessboard_blocks = masking_block(blocks)

        decoder_outputs_all.extend(decoder_outputs)
        spatial_mask_infos_all.extend(spatial_mask_infos)
        blocks_all.extend(chessboard_blocks)
        cnt += 1

    return np.array(blocks_all), np.array(decoder_outputs_all),np.array(spatial_mask_infos_all)


directory_name = r"D:\czp\RIDHproject\BOSSbase_1.01"
# test_model process
if __name__ == "__main__":
    encoder_input, decoder_output,spatial_mask_infos_all,origin_all = get_training_change_rate(directory_name, 10, 1000, 8)

    # print(encoder_input.shape)
    # print(decoder_output.shape)
    # print(spatial_mask_infos_all.shape)
    # print(origin_all[1000:1010])
    # print(encoder_input[1000:1010])
    # print("============================================")
    # print(decoder_output[1000:1010])
    # print(spatial_mask_infos_all[1000:1010])