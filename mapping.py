import math
import numpy as np


def make_dict():
    vestIdx = [0 for i in range(40)]
    return vestIdx


def mapping_function(z, c, d, w, h, vestIdx):
    w1, w2, w3, w4, w5, w6, w7, w8 = w/8, w/4, w*3/8, w/2, w*5/8, w*6/8, w*7/8, w
    h1, h2, h3, h4, h5 = h/5, h*2/5, h*3/5, h*4/5, h



    #di = math.sqrt((h * h) + (w * w))

    z = z/641 * 100


    if c <= w1 and d <= h1:
        vestIdx[0] += z
        return vestIdx
    elif w1 < c <= w2 and d <= h1:
        vestIdx[1] += z
        return vestIdx
    elif w2 < c <= w3 and d <= h1:
        vestIdx[2] += z
        return vestIdx
    elif w3 < c <= w4 and d <= h1:
        vestIdx[3] += z
        return vestIdx
    elif w4 < c <= w5 and d <= h1:
        vestIdx[4] += z
        return vestIdx
    elif w5 < c <= w6 and d <= h1:
        vestIdx[5] += z
        return vestIdx
    elif w6 < c <= w7 and d <= h1:
        vestIdx[6] += z
        return vestIdx
    elif w7 < c <= w8 and d <= h1:
        vestIdx[7] += z
        return vestIdx

    elif c <= w1 and h1 < d <= h2:
        vestIdx[8] += z
        return vestIdx
    elif w1 < c <= w2 and h1 < d <= h2:
        vestIdx[9] += z
        return vestIdx
    elif w2 < c <= w3 and h1 < d <= h2:
        vestIdx[10] += z
        return vestIdx
    elif w3 < c <= w4 and h1 < d <= h2:
        vestIdx[11] += z
        return vestIdx
    elif w4 < c <= w5 and h1 < d <= h2:
        vestIdx[12] += z
        return vestIdx
    elif w5 < c <= w6 and h1 < d <= h2:
        vestIdx[13] += z
        return vestIdx
    elif w6 < c <= w7 and h1 < d <= h2:
        vestIdx[14] += z
        return vestIdx
    elif w7 < c <= w8 and h1 < d <= h2:
        vestIdx[15] += z
        return vestIdx

    elif c <= w1 and h2 < d <= h3:
        vestIdx[16] += z
        return vestIdx
    elif w1 < c <= w2 and h2 < d <= h3:
        vestIdx[17] += z
        return vestIdx
    elif w2 < c <= w3 and h2 < d <= h3:
        vestIdx[18] += z
        return vestIdx
    elif w3 < c <= w4 and h2 < d <= h3:
        vestIdx[19] += z
        return vestIdx
    elif w4 < c <= w5 and h2 < d <= h3:
        vestIdx[20] += z
        return vestIdx
    elif w5 < c <= w6 and h2 < d <= h3:
        vestIdx[21] += z
        return vestIdx
    elif w6 < c <= w7 and h2 < d <= h3:
        vestIdx[22] += z
        return vestIdx
    elif w7 < c <= w8 and h2 < d <= h3:
        vestIdx[23] += z
        return vestIdx

    elif c <= w1 and h3 < d <= h4:
        vestIdx[24] += z
        return vestIdx
    elif w1 < c <= w2 and h3 < d <= h4:
        vestIdx[25] += z
        return vestIdx
    elif w2 < c <= w3 and h3 < d <= h4:
        vestIdx[26] += z
        return vestIdx
    elif w3 < c <= w4 and h3 < d <= h4:
        vestIdx[27] += z
        return vestIdx
    elif w4 < c <= w5 and h3 < d <= h4:
        vestIdx[28] += z
        return vestIdx
    elif w5 < c <= w6 and h3 < d <= h4:
        vestIdx[29] += z
        return vestIdx
    elif w6 < c <= w7 and h3 < d <= h4:
        vestIdx[30] += z
        return vestIdx
    elif w7 < c <= w8 and h3 < d <= h4:
        vestIdx[31] += z
        return vestIdx

    elif c <= w1 and h4 < d <= h5:
        vestIdx[32] += z
        return vestIdx
    elif w1 < c <= w2 and h4 < d <= h5:
        vestIdx[33] += z
        return vestIdx
    elif w2 < c <= w3 and h4 < d <= h5:
        vestIdx[34] += z
        return vestIdx
    elif w3 < c <= w4 and h4 < d <= h5:
        vestIdx[35] += z
        return vestIdx
    elif w4 < c <= w5 and h4 < d <= h5:
        vestIdx[36] += z
        return vestIdx
    elif w5 < c <= w6 and h4 < d <= h5:
        vestIdx[37] += z
        return vestIdx
    elif w6 < c <= w7 and h4 < d <= h5:
        vestIdx[38] += z
        return vestIdx
    elif w7 < c <= w8 and h4 < d <= h5:
        vestIdx[39] += z
        return vestIdx
    else:
        return vestIdx


