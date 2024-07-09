"""
@Function:  调用matlab
@Author : ZhangPeiCheng
@Time : 2024/2/16 17:24
"""

import matlab.engine
import numpy as np


#matlab_file_path = "./testcode.m"


def call_function_with_arr(dir_name,func_name,nargout, *args):
    """
    带参数的matlab方法封装(也可以执行不带参数的)
    :param nargout: 函数参数返回个数
    :param dir_name: 方法所在目录
    :param func_name:函数名称
    :param args:参数（可传入多个参数）
    :return:
    """
    eng = matlab.engine.start_matlab()
    eng.addpath(dir_name)
    # 这个地方标注一下nargout参数是制定getattr这个函数会返回的参数数量限制
    # 如果这个参数写的数量和实际函数返回数量不同会报错
    # 如果不需要传递任何的参数就不需要输入*args
    if args:
        result = getattr(eng, func_name)(*args, nargout=nargout)
        # 这个函数如果修改了返回值输入，这个地方要重新设置nargout参数
    else:
        result = getattr(eng, func_name)(nargout=nargout)
    eng.quit()
    return result


if __name__ == "__main__":
    dir_path = r'D:\czp\mae\call_matlab'
    #x = call_function_with_arr(dir_path,'testcode',0)
    #y = call_function_with_arr(dir_path,'testcode2',1,10,20)
    avail_pix_num = 32
    N = 32
    K = 16
    crossover_p = 0.1
    [x,y] = call_function_with_arr(dir_path, 'polar_encode_python', 2, avail_pix_num, N,K,crossover_p)
    z = call_function_with_arr(dir_path, 'polar_decode_python', 1, y, N,K,crossover_p)

    print("testcode call result: "+str(x))
    print("testcode2 call result: "+str(y))
    print("testcode2 call result: " + str(z))