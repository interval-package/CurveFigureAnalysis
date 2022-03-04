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

        # preprocess, get blurred gray scale pic
        def process2Gray(pic):
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            return gray

        self.gray = process2Gray(self.rawPic)

        if testVersion:
            self.main()
        pass

    def getMask(self):
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
        return maskArea

    def GetColorInterval(self, channel=0, LineCloNums=2, distance=20):
        # 对采样图像，进行颜色分析
        src = cv2.GaussianBlur(self.rawPic, (3, 3), 0)
        if channel < 0:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            channel = 0
        # 这里要将histRange设置为[0,256]，才会将空白计算进去，这是一个右开区间，同histSize=[256]
        # 同时，只对图像的主部分进行取样，不对边界进行色域提取，掩膜使用原来定义的掩膜
        hist = cv2.calcHist([src], [channel], self.mask, [256], [0, 256])
        histSortedIndex = np.argsort(hist, 0)[::-1]
        # 像素量最大的为背景颜色，其次记为线的颜色
        backgroundColor = histSortedIndex[0]
        # 先设置线的颜色为次多的
        lineColors = [histSortedIndex[1]]
        # 限制背景颜色至少与曲线颜色差值足够大
        CloCount = 0
        for Clo in histSortedIndex[1:]:
            if abs(backgroundColor - Clo) > distance:
                lineColors.append(Clo)
            else:
                CloCount += 1
            if CloCount > LineCloNums:
                break
        return backgroundColor, lineColors

    def TotalFilter(self, distance=20):
        b, g, r = cv2.split(self.rawPic)
        binPics = []
        for pic, channel in zip((self.gray, b, g, r), range(-1, 3)):
            backClo, Clos = self.GetColorInterval(channel=channel)
            tempBin = None
            for Clo in Clos:
                if tempBin is None:
                    print(Clo - distance, Clo + int(distance))
                    tempBin = cv2.inRange(pic, int(Clo) - distance, int(Clo) + distance)
                elif isinstance(tempBin, np.ndarray):
                    tempBin = cv2.bitwise_and(tempBin, cv2.inRange(pic, Clo - distance, Clo + distance))
                else:
                    raise ValueError("cv2 cannot filter the pic by Clo:%d" % Clo)
            if tempBin is not None:
                tempBin[tempBin == backClo] = 0
                if backClo > 100:
                    tempBin = 255 - tempBin
                binPics.append(tempBin)
            else:
                raise ValueError("work undone, the tempBin still None")
        return binPics

    def ColorHistCalc(self):
        hist_0 = cv2.calcHist([self.rawPic], [0], self.mask, [256], [0, 255]).reshape(256)  # open distinct
        hist_1 = cv2.calcHist([self.rawPic], [1], self.mask, [256], [0, 255]).reshape(256)
        hist_2 = cv2.calcHist([self.rawPic], [2], self.mask, [256], [0, 255]).reshape(256)
        hist_gray = cv2.calcHist([self.gray], [0], self.mask, [256], [0, 255]).reshape(256)
        x = np.arange(0, 256)
        return hist_0, hist_1, hist_2, hist_gray, x

    def imgOverlay(self):
        gray, b, g, r = self.TotalFilter()
        return cv2.bitwise_or(b, cv2.bitwise_or(g, r))

    def main(self):
        pic = self.gray
        pic = cv2.bitwise_and(pic, pic, mask=self.mask)
        gray, b, g, r = self.TotalFilter()
        plt.subplot(2, 2, 1)
        plt.imshow(gray, 'gray')
        plt.subplot(2, 2, 2)
        plt.imshow(b, 'gray')
        plt.subplot(2, 2, 3)
        plt.imshow(g, 'gray')
        plt.subplot(2, 2, 4)
        plt.imshow(r, 'gray')
        plt.show()
        pass


if __name__ == '__main__':
    for id in range(20, 100):
        LineFigure("../../data/img_train_BrokenLine/%d" % id, True)
    pass
