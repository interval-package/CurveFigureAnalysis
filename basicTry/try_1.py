import abc
import warnings
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import singledispatch


def show(inputs: list):
    l = len(inputs)
    n = math.ceil(math.sqrt(l))
    fig = plt.figure("compiled show")
    for i in range(l):
        plt.subplot(n, n, i + 1), plt.imshow(inputs[i], 'gray'), plt.axis('off')
    return fig


def GetMask(rawPic):
    rows, cols = rawPic.shape[:2]
    maskArea = np.zeros([rows, cols], dtype=np.uint8)
    # x, y, w, h = self.box.rectBox[1:]
    # maskArea[int(y):int(y+h),int(x):int(x+w)] = 255
    maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
    return maskArea


def DetectPointHarrisMethod(src: np.ndarray, gap=0.01, mask=None, inplace=False):
    img = src.copy()
    if src.ndim > 2:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

    dst = cv2.cornerHarris(src=img, blockSize=2, ksize=3, k=0.04)
    ret, dst = cv2.threshold(dst, dst.max() * gap, 255, 0)
    dst = np.uint8(dst)
    y, x = np.where(dst > 0)
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    if inplace:
        # src[x, y] = [0, 0, 255]
        for i in range(0, len(x) - 1):
            cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=[0, 0, 255])
    return x, y


def DetectPointGoodFeatureMethod(src: np.ndarray, gap=0.01, maxCount=100, mask=None, inplace=False):
    img = src.copy()
    if src.ndim > 2:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
    corners = cv2.goodFeaturesToTrack(img, maxCount, gap, 100)
    corners = np.int0(corners).reshape(corners.shape[0], corners.shape[2])
    corners = corners[np.argsort(corners[:, 0]), :]
    if inplace:
        for i in range(0, corners.shape[0] - 1):
            print((corners[i][1], corners[i][0]), (corners[i + 1][1], corners[i + 1][0]))
            cv2.line(src, (corners[i][0], corners[i][1]), (corners[i + 1][0], corners[i + 1][1]), color=[0, 255, 0])
    return corners


def linePlotor(src, points, color=None):
    if color is None:
        color = [0, 0, 255]
    if isinstance(points, tuple):
        x, y = points
    elif isinstance(points, np.ndarray):
        x = points[:, 0]
        y = points[:, 1]
    else:
        x = []
        y = []
        raise ValueError("unfitted input")
    for i in range(0, len(x) - 1):
        cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=color)
    return src


def main(id: int):
    src = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
    maskpic_target = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)

    showPics = [src, maskpic_target]

    show(showPics)
    plt.show()


if __name__ == '__main__':
    for i in range(1, 100, 5):
        main(i)
    pass
