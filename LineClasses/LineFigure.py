import abc

import cv2
import numpy as np

from utils.picProcessors import readPicFromFile

from LineClasses.HEDMethod.HEDRun import HEDDetect, PicTrans2HEDInput


class LineFigure(object):

    # constructors
    @abc.abstractmethod
    def __init__(self, rawPic, givenPic=None, picLabel=None):
        """
        :param rawPic:
        :param givenPic: the given pic, maybe None
        :param picLabel:
        """
        # testVersion：是否以测试模式构建对象
        if isinstance(rawPic, str):
            raise TypeError("this method is dealt with np.ndarray get str instead, "
                            "please use the class Method @fromFile for str path ")
        # self.testVersion = testVersion
        self.rawPic, givenPic, self.picLabel = PicTrans2HEDInput(rawPic), PicTrans2HEDInput(givenPic), picLabel
        if givenPic is not None:
            _, givenPic = cv2.threshold(cv2.cvtColor(givenPic, cv2.COLOR_BGR2GRAY), 125, 255, cv2.THRESH_BINARY)
        self.givenPic = givenPic
        self.mask = self.getMask()

        # preprocess, get blurred gray scale pic
        def process2Gray(pic):
            gray = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            return gray

        self.gray = process2Gray(self.rawPic)

        self.processedPic = None
        self.bin_set = None

        # if testVersion:
        #     self.main()
        pass

    @classmethod
    def fromFile(cls, basicPath: str):
        """
        :param basicPath: the folder containing the pics
        """
        rawPic, binaryPic, picLabel = readPicFromFile(basicPath)
        return cls(rawPic, binaryPic, picLabel)

    # utils
    @staticmethod
    def IsBinPic_Valid(pic: np.ndarray, gap=1000) -> bool:
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

    def IsBinPic_GoodQuality(self):
        pass

    def getMask(self) -> np.ndarray:
        """
        :return 通过设定的预期值获取最基本的蒙版
        """
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(rows * 0.125):int(rows * 0.875), int(62):int(435)] = 255
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

    def BinPic_TotalFilter(self, distance=10):
        """
        对三个色道，以及灰度的图片进行基于色值分布曲线的提取
        :param distance:　for each filtered clo we define a gap for each to in range
        :return :extraction of three Channel plus gray figure
        """
        # 对三个色道，以及灰度的图片进行基于色值分布曲线的提取
        # update：使用hsv比使用bgr好
        h, s, v = cv2.split(cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2HSV))
        binPics = []

        for pic, channel in zip((self.gray, h, s, v), range(-1, 3)):
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
                if self.IsBinPicNormalized(tempBin):
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

    def IsBinPicNormalized(self, pic_in) -> bool:
        hist_inner = cv2.calcHist([pic_in], [0], self.mask, [2], [0, 256])
        return hist_inner[0] < hist_inner[-1]

    # BinPic_ methods render the binary output

    def BinPic_getCannyPic(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        cannyPic = cv2.dilate(cv2.Canny(self.gray, threshold1=5, threshold2=5), kernel)
        cannyPic = cv2.erode(cannyPic, kernel)
        if self.IsBinPicNormalized(cannyPic):
            cannyPic = 255 - cannyPic
        cannyPic = cv2.bitwise_and(cannyPic, self.mask)
        return cannyPic

    def BinPic_AdaptiveThresh(self):
        threshPic = cv2.adaptiveThreshold(src=self.gray, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY_INV, blockSize=11, C=12)
        if self.IsBinPicNormalized(threshPic):
            threshPic = 255 - threshPic
        threshPic = cv2.bitwise_and(threshPic, self.mask)
        return threshPic

    def BinPic_SetGetter(self):
        gray, h, s, v = self.BinPic_TotalFilter()

        threshPic = self.BinPic_AdaptiveThresh()
        cannyPic = self.BinPic_getCannyPic()

        # hed = self.BinPic_HEDMethod_Adapt_Tres()
        # processed_hed = self.BinPic_HEDMethod_Processed()

        bin_set = [threshPic, gray, v, cannyPic]

        self.bin_set = bin_set

        return bin_set

    # bin pic output
    def BinPic_imgOverlay(self):
        result = None
        bin_set = self.BinPic_SetGetter()
        # 这里使用的是全验证方式
        for pic in bin_set:
            if self.IsBinPic_Valid(pic):
                if result is None:
                    result = pic
                else:
                    result = cv2.bitwise_and(result, pic)
        if result is None:
            raise ValueError("img overlay failed, with return None")

        if self.processedPic is None:
            self.processedPic = result
        return result

    def BinPic_SmoothOutput(self):
        if self.processedPic is None:
            result = self.BinPic_imgOverlay()
        else:
            result = self.processedPic
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        result = cv2.dilate(result, kernel)
        return result

    # undone
    def BinPic_CentralDivide(self, precision=0.1, epoch=100):
        # 动态阈值算法
        h, s, v = cv2.split(cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2HSV))
        temp = []
        for channel in [h, s, v]:
            # 随机初始化的阈值
            thresh = 125
            # 备份通道图片用于后续迭代
            tempPic = channel.copy()
            i = 0
            while i < epoch:
                i += 1
                tempPic = cv2.threshold(channel, thresh, 255, cv2.THRESH_BINARY)
        return

    #################################################################################

    # super func hed method
    def BinPic_HEDMethod_Processed(self):
        pic = HEDDetect(self.rawPic)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        _, res = cv2.threshold(pic, 40, 255, cv2.THRESH_BINARY)
        res = cv2.dilate(res, kernel, iterations=1)
        res = cv2.erode(res, kernel, iterations=1)
        res = np.bitwise_and(res, self.mask)
        return res

    def BinPic_HEDMethod_Adapt_Tres(self):
        pic = HEDDetect(self.rawPic)
        pic = np.bitwise_and(pic, self.mask)
        _, pic = cv2.threshold(pic, 10, 255, cv2.THRESH_BINARY)
        return pic


if __name__ == '__main__':
    for id in range(32, 100):
        print(id)
        LineFigure.fromFile("../../data/img_train_BrokenLine/%d" % id)
    pass
