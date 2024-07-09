"""
@Function:  build mae + conv2d train set
@Author : ZhangPeiCheng
@Time : 2023/12/10 11:43
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


def get_mask_surround_pixel(block, index):
    # seq: [up, down, left, right]
    i, j = index
    # center
    n, m = block.shape
    # have many situations
    # 4 corners (i == n-1 and j == 0) or (i == 0 and j == m-1) or (i == n-1 and j == m-1)
    if i == 0 and j == 0:
        mask_surround_pixel = [-1, block[i + 1, j], -1, block[i, j + 1]]
        return mask_surround_pixel
    if i == 0 and j == m - 1:
        mask_surround_pixel = [-1, block[i + 1, j], block[i, j - 1], -1]
        return mask_surround_pixel
    if i == n - 1 and j == 0:
        mask_surround_pixel = [block[i - 1, j], -1, -1, block[i, j + 1]]
        return mask_surround_pixel
    if i == n - 1 and j == m - 1:
        mask_surround_pixel = [block[i - 1, j], -1, block[i, j - 1], -1]
        return mask_surround_pixel
    # up
    if i == 0:
        mask_surround_pixel = [-1, block[i + 1, j], block[i, j - 1], block[i, j + 1]]
        return mask_surround_pixel
    # bottom
    if i == n-1:
        mask_surround_pixel = [block[i - 1, j], -1, block[i, j - 1], block[i, j + 1]]
        return mask_surround_pixel
    # left
    if j == 0:
        mask_surround_pixel = [block[i - 1, j], block[i + 1, j], -1, block[i, j + 1]]
        return mask_surround_pixel
    # right
    if j == m-1:
        mask_surround_pixel = [block[i - 1, j], block[i + 1, j], block[i, j - 1], -1]
        return mask_surround_pixel
    # center
    mask_surround_pixel = [block[i - 1, j], block[i + 1, j], block[i, j - 1], block[i, j + 1]]
    return mask_surround_pixel

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
            is_odd_column = (j + 1) % 2 == 1
            if (is_odd_row and is_odd_column) or (not is_odd_row and not is_odd_column):
                # 第一个点是加密像素
                decoder_output.append(image_block[i, j])
                mask_surround_pixel = get_mask_surround_pixel(image_block,(i, j))
                mask_surround_pixels.append(mask_surround_pixel)
            else:
                chessboard_blocks[i, j] = image_block[i, j]
                encoder_input.append(image_block[i, j])

    return np.array(encoder_input), chessboard_blocks, decoder_output,np.array(mask_surround_pixels)


def masking_block(blocks):
    """
    Masking operations on blocks
    Now, let's start by dividing the board into squares
    => Subsequent changes can be made to randomized masks with mask rate adjustments
    """
    encoder_inputs = []
    chessboard_blocks = []
    spatial_mask_infos = []
    decoder_outputs = []
    for block in blocks:
        encoder_input, chessboard_block, decoder_output, spatial_mask_info = create_chessboard_blocks(block)
        encoder_inputs.append(encoder_input)
        chessboard_blocks.append(chessboard_block)
        spatial_mask_infos.append(spatial_mask_info)
        decoder_outputs.append(decoder_output)
    return  np.array(decoder_outputs), np.array(spatial_mask_infos),np.array(chessboard_blocks)
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
def get_training_set(directory, pic_cnt, block_cnt, block_size):
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
        # blocks, block_indexes = random_block_split(pic, block_size, block_cnt)
        blocks, block_indexes = block_split(pic, block_size)
        # 随机选择块
        blocks, block_indexes = random_block_split_for_all(blocks, block_cnt, block_indexes)
        encoder_inputs, decoder_outputs, spatial_mask_infos,chessboard_blocks = masking_block(blocks)
        decoder_outputs_all.extend(decoder_outputs)
        spatial_mask_infos_all.extend(spatial_mask_infos)
        blocks_all.extend(chessboard_blocks)
        cnt += 1
    return np.array(blocks_all), np.array(decoder_outputs_all),np.array(spatial_mask_infos_all)

def get_training_for_pic(directory, pic_cnt):
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
        encoder_inputs, decoder_outputs, spatial_mask_infos,chessboard_blocks = masking_block(blocks)
        decoder_outputs_all.extend(decoder_outputs)
        spatial_mask_infos_all.extend(spatial_mask_infos)
        blocks_all.extend(chessboard_blocks)
        cnt += 1
    return np.array(blocks_all), np.array(decoder_outputs_all),np.array(spatial_mask_infos_all)


directory_name = r"D:\czp\RIDHproject\BOSSbase_1.01"
# test_model process
if __name__ == "__main__":
    encoder_input, decoder_output,spatial_mask_infos_all = get_training_set(directory_name, 100, 1000, 6)
    print(encoder_input.shape)
    print(decoder_output.shape)
    print(encoder_input[10000:10010])
    print("============================================")
    print(decoder_output[10000:10010])