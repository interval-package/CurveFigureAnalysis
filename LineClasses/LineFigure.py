import abc
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import singledispatch


def adpativeShow(inputs: list):
    l = len(inputs)
    n = math.ceil(math.sqrt(l))
    fig = plt.figure("compiled show")
    for i in range(l):
        plt.subplot(n, n, i + 1), plt.imshow(inputs[i], 'gray'), plt.axis('off')
    return fig


def eraseFrame(img: np.ndarray):
    if img.ndim > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret, binary = cv2.threshold(gray, 200, 255, 0)
    # assuming, b_w is the binary image
    inv = 255 - binary
    horizontal_img = inv
    vertical_img = inv

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
    vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

    mask_img = horizontal_img + vertical_img
    # no_border = np.bitwise_or(binary, mask_img)
    no_border = cv2.bitwise_or(binary, mask_img)

    return no_border


# 读取文件
def readLine(basicPath: str):
    # 标准化读取图片信息
    # print(basicPath + "/draw.png")
    try:
        rawpic = cv2.imread(basicPath + "/draw.png")
        binarypic = cv2.imread(basicPath + "/draw_mask.png")
        piclable = pd.read_table(basicPath + "/db.txt", engine='python', delimiter="\n")
        return rawpic, binarypic, piclable
    except IOError as IOe:
        print('repr(IOe):\t', repr(IOe))
        return np.array([]), np.array([]), []
    except Exception as e:
        print('repr(e):\t', repr(e))
        return np.ndarray([]), np.ndarray([]), []


# 由像素检测线上的点
def LinePointDetectCentralize(scrGray):
    # 提取线的中点坐标，将中点坐标输出
    if scrGray.ndim > 2:
        # if not gray, change into gray
        scrGray = cv2.cvtColor(scrGray, cv2.COLOR_BGR2GRAY)
    # 再次确认为二进制图片
    scrGray = cv2.threshold(scrGray, 200, 255, cv2.THRESH_BINARY)[1]

    # 这里进行了一次旋转，因为np.where的遍历是沿着行方向进行的
    idx_x, idx_y = np.where(cv2.rotate(scrGray, cv2.ROTATE_90_CLOCKWISE, 90))

    # 对x做一次差分，找出x有递增的点
    dx = np.diff(idx_x)
    # 获取递增点序号，并且在初始处插入一个0
    x_stage = np.insert(np.where(dx), 0, 0)

    # 现在1表示一串相同x的初始点
    x_gap = np.diff(x_stage) + 1
    x_gap = np.floor_divide(x_gap, 2)
    x_gap = np.append(x_gap, 0)
    x_ = x_gap + x_stage

    try:
        idx_x_sep = idx_x[x_]
        idx_y_sep = idx_y[x_]
    except IndexError:
        idx_x_sep = idx_x
        idx_y_sep = idx_y

    return idx_x, idx_y, idx_x_sep, idx_y_sep


class LineFigure(object):
    @abc.abstractmethod
    @singledispatch
    def __init__(self, basicPath: str, testVersion=False):
        # testVersion：是否以测试模式构建对象
        self.testVersion = testVersion
        self.rawPic, self.binaryPic, self.picLable = readLine(basicPath)

        self.mask = self.getMask()
        if testVersion:
            self.main()
        pass

    def getMask(self):
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
        return maskArea

    def ColorHistCalc(self):
        gray = cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
        hist_0 = cv2.calcHist([self.rawPic], [0], self.mask, [256], [0, 255]).reshape(256)  # open distinct
        hist_1 = cv2.calcHist([self.rawPic], [1], self.mask, [256], [0, 255]).reshape(256)
        hist_2 = cv2.calcHist([self.rawPic], [2], self.mask, [256], [0, 255]).reshape(256)
        hist_gray = cv2.calcHist([imgBlur], [0], self.mask, [256], [0, 255]).reshape(256)
        x = np.arange(0, 256)
        return hist_0, hist_1, hist_2, hist_gray, x

    def filterColorMethod(self, numBoundUp, numBoundDown=20):
        hist_0, hist_1, hist_2, hist_gray, x = self.ColorHistCalc()
        x_tar = []
        for hist in [hist_0, hist_1, hist_2, hist_gray]:
            x_tar.append(x[np.bitwise_and(hist > numBoundDown, hist < numBoundUp)])
        return x_tar

    def Cannal3Filter(self):
        x_tar = self.filterColorMethod(5000, 500)
        gray = cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(gray, (5, 5), 1)  # ADD GAUSSIAN BLUR
        pic = np.zeros(imgBlur.shape)
        for hist_x, cannal in zip(x_tar, range(3)):
            for colorValue in hist_x:
                pic[self.rawPic[:, :, cannal] == colorValue] = 255
        cv2.morphologyEx(pic, cv2.MORPH_OPEN, (5, 5))
        return pic

    def main(self):
        pic = self.Cannal3Filter()
        pic = cv2.bitwise_and(pic, pic, mask=self.mask)
        plt.subplot(1, 2, 1)
        plt.imshow(pic, 'gray')
        plt.subplot(1, 2, 2)
        plt.imshow(self.rawPic)
        plt.show()
        pass


if __name__ == '__main__':
    for id in range(20, 100):
        LineFigure("../../data/img_train_BrokenLine/%d" % id, True)
    pass
