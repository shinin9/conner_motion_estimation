import cv2
import numpy as np


def ui_vib(vestIdx):
    background = np.full((720, 1280, 3), 0, np.uint8)
    #cv2.line(background, (1280//4, 0), (1280//4, 720), (255, 255, 255))
    #cv2.line(background, (1280//4*3, 0), (1280//4*3, 720), (255, 255, 255))

    x = 1280//8//2
    y = 720//5//2
    for i in range(len(vestIdx)):
        if vestIdx[i] == 0:
            v = 255
            cv2.circle(background, (x, y), 50, (v, v, v))
        else:
            v = 255*vestIdx[i]/100
            cv2.circle(background, (x, y), 50, (v, v, v), -1)
        x += 1280 // 8
        if (i+1) % 8 == 0:
            x = 1280//8//2
            y = y + 720 // 5

    return background


