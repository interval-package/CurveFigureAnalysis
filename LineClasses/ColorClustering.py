import cv2
import numpy as np
import matplotlib.pyplot as plt


def GetMaskSimple(rawPic):
    rows, cols = rawPic.shape[:2]
    maskArea = np.zeros([rows, cols], dtype=np.uint8)
    # x, y, w, h = self.box.rectBox[1:]
    # maskArea[int(y):int(y+h),int(x):int(x+w)] = 255
    maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
    return maskArea


def movePic(pic, x=0, y=0):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols = pic.shape[0:2]
    res = cv2.warpAffine(pic, mat, (cols, rows))
    return res


class ClusteringNode(object):




if __name__ == '__main__':

    pass
