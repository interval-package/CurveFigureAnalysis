import abc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PointDetector import *
from utils.picProcessors import readLine


class LineFigure(object):

    @abc.abstractmethod
    def __init__(self, rawPic, givenPic=None, picLabel=None, testVersion=False):
        """
        :param rawPic:
        :param givenPic: the given pic, maybe None
        :param picLabel:
        :param testVersion: test or not, if test, do the main func
        """
        # testVersion：是否以测试模式构建对象
        if isinstance(rawPic, str):
            raise TypeError("this method is dealt with np.ndarray get str instead, "
                            "please use the class Method @fromFile for str path ")
        self.testVersion = testVersion
        self.rawPic, self.givePic, self.picLabel = rawPic, givenPic, picLabel
        self.mask = self.getMask()

        # preprocess, get blurred gray scale pic
        def process2Gray(pic):
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            return gray

        self.gray = process2Gray(self.rawPic)

        self.processedPic = None

        if testVersion:
            self.main()
        pass

    @classmethod
    @abc.abstractmethod
    def fromFile(cls, basicPath: str, testVersion=False):
        """
        :param basicPath: the folder containing the pics
        :param testVersion: test or not
        """
        rawPic, binaryPic, picLabel = readLine(basicPath)
        return cls(rawPic, binaryPic, picLabel, testVersion)

    @staticmethod
    def binPicCertification(pic: np.ndarray, gap=10000) -> bool:
        """
        :param pic: binary pic
        :param gap: the minimal num of valid points
        :return: bool, the pic is valid or not
        """
        # 图像有效性验证，如果图像内部为真的像素点过少，则验证失败
        if pic.ndim > 2:
            raise ValueError("you should input a binary pic for certification")
        hist_inner = cv2.calcHist([pic], [0], None, [2], [0, 256])
        # print(hist_inner)
        return hist_inner[-1] > gap

    def getMask(self) -> np.ndarray:
        """
        :return 通过设定的预期值获取最基本的蒙版
        """
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(rows * 0.125):int(rows * 0.875), int(cols * 0.127):int(cols * 0.9)] = 255
        return maskArea

    def GetColorInterval(self, channel=0, LineCloNums=2, distance=20):
        """
        对采样图像，进行颜色分析
        :param channel: filter the target channel color, -1 is the gray
        :param LineCloNums: nums of valid colors
        :param distance:　the linClo must be different from backGround Clo, the least Clo distance between
        """
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

    def TotalFilter(self, distance=10):
        """
        对三个色道，以及灰度的图片进行基于色值分布曲线的提取
        :param distance:　for each filtered clo we define a gap for each to in range
        :return :extraction of three Channel plus gray figure
        """
        # 对三个色道，以及灰度的图片进行基于色值分布曲线的提取
        b, g, r = cv2.split(self.rawPic)
        binPics = []

        # 将图片标准化，白色背景的图片将会返回True，由后续反转颜色
        def BinPicNormalize(pic_in) -> bool:
            hist_inner = cv2.calcHist([pic_in], [0], self.mask, [2], [0, 256])
            return hist_inner[0] < hist_inner[-1]

        for pic, channel in zip((self.gray, b, g, r), range(-1, 3)):
            backClo, Clos = self.GetColorInterval(channel=channel)
            tempBin = None
            # 色彩分析会得到多种颜色，对每一种颜色进行筛选
            for Clo in Clos:
                if tempBin is None:
                    tempBin = cv2.inRange(pic, int(Clo) - distance, int(Clo) + distance)
                elif isinstance(tempBin, np.ndarray):
                    # 将每一种颜色筛选结果进行合并
                    tempBin = cv2.bitwise_and(tempBin, cv2.inRange(pic, Clo - distance, Clo + distance))
                else:
                    raise ValueError("cv2 cannot filter the pic by Clo:%d" % Clo)
            if tempBin is not None:
                if BinPicNormalize(tempBin):
                    # 反转颜色
                    tempBin = 255 - tempBin
                binPics.append(cv2.bitwise_and(tempBin, self.mask))
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

    def imgOverlay(self) -> np.ndarray:
        result = None

        gray, b, g, r = self.TotalFilter()
        thresPic = cv2.bitwise_and(
            cv2.adaptiveThreshold(src=self.gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=12),
            self.mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        cannyPic = cv2.dilate(cv2.Canny(self.gray, threshold1=5, threshold2=5), kernel)
        cannyPic = cv2.bitwise_and(cannyPic, self.mask)
        for pic in [gray, b, g, r, thresPic, cannyPic]:
            if self.binPicCertification(pic):
                if result is None:
                    result = pic
                else:
                    result = cv2.bitwise_and(result, pic)
        if result is None:
            raise ValueError("img overlay failed, with return None")

        if self.processedPic is None:
            self.processedPic = result
        return result

    def smoothOutput(self):
        if self.processedPic is None:
            result = self.imgOverlay()
        else:
            result = self.processedPic
        # if ~self.binPicCertification(result, 20000):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        result = cv2.dilate(result, kernel)
        # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        # result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        return result

    def main(self):
        gray, b, g, r = self.TotalFilter()
        threshPic = cv2.bitwise_and(
            cv2.adaptiveThreshold(src=self.gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=12),
            self.mask)
        cannyPic = cv2.bitwise_and(
            cv2.Canny(self.gray, threshold1=5, threshold2=5),
            self.mask)
        pics = [gray, b, g, r, threshPic, cannyPic, self.rawPic, self.imgOverlay(), self.smoothOutput()]
        plt.figure("1")
        for i, pic in zip(range(1, len(pics) + 1), pics):
            plt.subplot(3, 3, i)
            plt.imshow(pic, 'gray')
        plt.show()
        pass


if __name__ == '__main__':
    for id in range(32, 100):
        print(id)
        LineFigure.fromFile("../../data/img_train_BrokenLine/%d" % id, True)
    pass
