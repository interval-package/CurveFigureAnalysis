from functools import singledispatch
import matplotlib.pyplot as plt
from LineFigure import LineFigure
import numpy as np
import cv2


def DetectPointHarrisMethod(src: np.ndarray, gap=0.01, mask=None, inplace=False):
    src = src.copy()
    if src.ndim > 2:
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        img = src.copy()

    # imgBlur = cv2.GaussianBlur(img, (5, 5), 1)  # ADD GAUSSIAN BLUR
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    # img = cv2.erode(imgCanny, kernel, iterations=2)
    # img = cv2.dilate(img, kernel, iterations=2)
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11,
    #                                2)  # 自适应阈值化

    # 设置蒙版
    thresh = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)

    dst = cv2.cornerHarris(src=thresh, blockSize=2, ksize=3, k=0.04)
    # 通过阈值来得到Harris结果的有效点
    ret, dst = cv2.threshold(dst, dst.max() * gap, 255, 0)
    dst = np.uint8(dst)
    y, x = np.where(dst > 0)
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]
    if inplace:
        # src[x, y] = [0, 0, 255]
        for i in range(0, len(x) - 1):
            cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=[0, 0, 255])
    return x, y, src, thresh


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


def LinePointsPlot(src, points, color=None, PotType='line'):
    if color is None:
        color = [125, 125, 125]
    if isinstance(points, tuple):
        x, y = points
    elif isinstance(points, np.ndarray):
        x = points[:, 0]
        y = points[:, 1]
    else:
        raise ValueError("unfitted input")
    # 注意，这里图像要flip上下颠倒两次，才会画出正常的图像
    src = cv2.flip(src, 0)
    if PotType == 'line':
        for i in range(0, len(x) - 1):
            cv2.line(src, (x[i], y[i]), (x[i + 1], y[i + 1]), color=color, thickness=5)
    elif PotType == 'circle':
        for i in range(len(x)):
            cv2.circle(src, (x[i], y[i]), 5, color=color)
    elif PotType == 'dot':
        for i in range(len(x)):
            cv2.drawMarker(src, (x[i], y[i]), color=color, markerType=cv2.MARKER_CROSS, markerSize=2)
    else:
        raise ValueError("invalid PotType")
    # 返回时再次颠倒
    return cv2.flip(src, 0)


class BrokenLineFigure(LineFigure):
    @singledispatch
    def __init__(self, rawPic, givenPic=None, picLabel=None, testVersion=False):
        # read in the pic into the obj
        super().__init__(rawPic, givenPic, picLabel, testVersion)
        # super(BrokenLineFigure, self).__init__("../../data/img_train_BrokenLine/%d" % id, test)
        self.type = "BrokenLine"

    @classmethod
    def fromFile(cls, id: int, testVersion=False):
        rawPic, givenPic, picLabel = super().readLine("../../data/img_train_BrokenLine/%d" % id)
        return cls(rawPic, givenPic, picLabel, testVersion)

    def TurningPointGet(self):
        if self.processedPic is not None:
            pic = self.processedPic
        else:
            pic = self.smoothOutput()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pic = cv2.morphologyEx(pic, cv2.MORPH_BLACKHAT, kernel)
        return pic

    def getPoints(self):
        if self.processedPic is not None:
            pic = self.processedPic
        else:
            pic = self.smoothOutput()
        # 中心法获取图像点
        x, y, x_c, y_c = LineFigure.LinePointDetectCentralize(pic)
        # Harris角点检测
        x_harris, y_harris, _, _ = DetectPointHarrisMethod(pic)
        plt.subplot(2, 2, 1)
        # plt.plot(x_c, y_c, 'g')
        plt.imshow(self.TurningPointGet())
        plt.subplot(2, 2, 2)
        plt.imshow(LinePointsPlot(self.rawPic, (x, y), PotType='dot'))
        plt.subplot(2, 2, 3)
        plt.imshow(LinePointsPlot(self.rawPic, (x_c, y_c)))
        plt.subplot(2, 2, 4)
        plt.imshow(self.processedPic, 'gray')
        plt.show()


if __name__ == '__main__':
    for i in range(20, 30):
        obj = BrokenLineFigure.fromFile(i)
        obj.getPoints()
    pass
