import abc
import warnings
import cv2
# import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from FigureBox import FigureBox
from functools import singledispatch


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
    # 构造函数
    @abc.abstractmethod
    @singledispatch
    def __init__(self, basicPath: str, testVersion=False):
        # testVersion：是否以测试模式构建对象
        self.testVersion = testVersion
        try:
            self.rawPic, self.binaryPic, self.picLable = readLine(basicPath)
        except IOError as e:
            warnings.warn(repr(e))
            exit()
        self.box = FigureBox(self)
        self.mask = self.GetMask()
        self.processedPic = self.ProcessSelf()
        # 使用自己读取的二进制图像进行分析
        if self.testVersion:
            # self.binaryPic = self.SideFilter(self.processedPic, self.GetColorBoundry()[1], self.mask,
            # self.testVersion)
            self.binaryPic = cv2.add(self.processedPic,
                                     np.zeros(np.shape(self.processedPic), dtype=np.uint8),
                                     mask=self.mask)
            # self.box.disp()
        pass

    @__init__.register(np.ndarray)
    # 或者以图片形式传进来
    def __init__(self, pic: np.ndarray, testVersion=False):
        # testVersion：是否以测试模式构建对象
        self.testVersion = testVersion
        self.rawPic = pic
        self.picLable = ''
        self.binaryPic = self.ImageOverlay()
        self.mask = self.GetMask()
        self.box = FigureBox(self)
        self.processedPic = self.ProcessSelf()

    # 对自身图像进行预先部分处理
    def ProcessSelf(self):
        img = cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2GRAY)
        # kernel = np.ones((5,3), np.uint8)
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
        img = eraseFrame(img)
        img = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
        return img

    # 获取掩膜
    def GetMask(self):
        rows, cols = self.rawPic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        # x, y, w, h = self.box.rectBox[1:]
        # maskArea[int(y):int(y+h),int(x):int(x+w)] = 255
        maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
        return maskArea

    # 输入二进制图片，制作边缘掩膜
    @staticmethod
    def GetShapeMask(pic: np.ndarray, ksize=(13, 13)):
        # rows, cols = pic.shape[:2]
        # maskArea = np.zeros([rows, cols], dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
        maskArea = cv2.dilate(pic, kernel, iterations=1)
        maskArea = cv2.threshold(maskArea, 100, 255, cv2.THRESH_BINARY)[1]
        return maskArea

    # 通过自适应阈值过滤图片
    def AdaptiveThresholdPic(self):
        gray = cv2.cvtColor(self.rawPic, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, 1, 1, 11, 2)
        thresh = cv2.add(thresh, np.zeros(np.shape(thresh), dtype=np.uint8), mask=self.mask)
        return thresh

    # 对图片进行边缘检测，并获得边缘检测结果的二进制图像
    def getSobelPic(self):
        img = self.rawPic
        # Convert to graycsale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X

        sobel = [sobelx, sobely, sobelxy]
        result = []
        for so in sobel:
            tempSobel = cv2.normalize(so, dst=so.shape, mask=self.mask) * 1000
            thres, tempSobel = cv2.threshold(tempSobel.astype(np.uint8), 230, 255, cv2.THRESH_BINARY)
            result.append(tempSobel)
        # sobelx = cv2.add(sobelx, np.zeros(np.shape(sobelx), dtype=np.uint8), mask=self.mask)

        return result

    # 图像叠加，将内部所有的二进制图片作为掩膜对图像进行筛选
    def ImageOverlay(self):
        sobel = self.getSobelPic()
        sobel.append(self.AdaptiveThresholdPic())
        temp = np.ndarray([])
        for so, idx in zip(sobel, range(len(sobel))):
            if idx == 0:
                temp = so
            else:
                temp = cv2.add(temp, np.zeros(np.shape(temp), dtype=np.uint8), mask=self.GetShapeMask(so))
        return temp

    def GetColorBoundry(self):
        # 对采样图像，进行颜色分析
        src = self.processedPic
        if src.ndim > 2:
            # if not gray, change into gray
            src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # 这里要将histRange设置为[0,256]，才会将空白计算进去，这是一个右开区间，同histSize=[256]
        # 同时，只对图像的主部分进行取样，不对边界进行色域提取，掩膜使用原来定义的掩膜
        hist = cv2.calcHist([src], [0], self.mask, [256], [0, 256])
        histSortedIndex = np.argsort(hist, 0)[::-1]
        # 像素量最大的为背景颜色，其次记为线的颜色
        backgroundColor = histSortedIndex[0]
        iter = 1
        lineColor = histSortedIndex[iter]
        while abs(backgroundColor - lineColor) < 20:
            iter += 1
            lineColor = histSortedIndex[iter]
        # 打印统计信息
        if self.testVersion:
            print("Gray type: \n background color value: %d \n line color value: %d"
                  % (backgroundColor, lineColor))
            print(histSortedIndex[:5])
        return backgroundColor, lineColor, hist

    @staticmethod
    def SideFilter(image: np.ndarray, boundry: int, mask=None, testVersion=False, ksize=(2, 2)):
        # 过滤图像像素
        # image should be gray image
        if image.ndim > 2:
            # if not gray, change into gray
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 范围筛选
        img = cv2.inRange(image, boundry - 15, boundry + 15)
        # # 自适应筛选
        # img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        if testVersion:
            kernel = np.ones(ksize, np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel=kernel)
        img = cv2.add(img, np.zeros(np.shape(img), dtype=np.uint8), mask=mask)
        return img.astype(np.uint8)

    # 由像素检测线上的点
    def LinePointDetectCentralize(self):
        # 提取线的中点坐标，将中点坐标输出
        scrGray = self.binaryPic
        if scrGray.ndim > 2:
            # if not gray, change into gray
            scrGray = cv2.cvtColor(scrGray, cv2.COLOR_BGR2GRAY)

        scrGray = cv2.threshold(scrGray, 200, 255, cv2.THRESH_BINARY)[1]
        # 这里进行了一次旋转，因为np.where的遍历是沿着y行方向进行的
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

    # 计算图像色彩内容
    def CalcSelf(self):
        hist_0 = cv2.calcHist([self.rawPic], [0], self.mask, [256], [0, 256])
        hist_1 = cv2.calcHist([self.rawPic], [1], self.mask, [256], [0, 256])
        hist_2 = cv2.calcHist([self.rawPic], [2], self.mask, [256], [0, 256])
        return hist_0, hist_1, hist_2

    def show(self):
        # 每个show都会带来一个窗体
        h1 = self.show_1()
        h2 = self.__show_2()
        h3 = self.__show_3()
        plt.show()

    # 展示
    def show_1(self, slogan="Line Figure"):
        handle=plt.figure(slogan)
        if self.testVersion:
            plt.subplot(2, 2, 1), plt.imshow(self.processedPic, 'gray'), plt.axis('off')
        else:
            plt.subplot(2, 2, 1), plt.imshow(self.rawPic, 'gray'), plt.axis('off')
        plt.subplot(2, 2, 2)
        x, y, x_s, y_s = self.LinePointDetectCentralize()
        plt.plot(x, y, 'b.')
        plt.plot(x_s, y_s, 'r-')
        plt.subplot(2, 2, 3), plt.imshow(self.binaryPic, 'gray'), plt.axis('off')

        if self.testVersion:
            plt.subplot(2, 2, 4), plt.imshow(self.rawPic)
            plt.plot(x_s, self.rawPic.shape[0] - y_s, 'b-')
            plt.axis('off')
        else:
            hist = self.GetColorBoundry()[2]
            hist_0, hist_1, hist_2 = self.CalcSelf()
            rowSlicedCount = 8
            plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 1))), plt.plot(hist_0, 'b'), plt.title(
                "color hist b")
            plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 2))), plt.plot(hist_1, 'g'), plt.title(
                "color hist g")
            plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 3))), plt.plot(hist_2, 'r'), plt.title(
                "color hist r")
            plt.subplot(rowSlicedCount, 2, int(2 * (rowSlicedCount / 2 + 4))), plt.plot(hist), plt.title("gray hist")

        return handle

    def __show_2(self):
        handle = plt.figure("show_2")
        sobel = self.getSobelPic()
        plt.subplot(2, 2, 1)
        plt.imshow(self.ImageOverlay(), 'gray'), plt.axis('off')
        for i in range(3):
            try:
                plt.subplot(2, 2, 2 + i)
                plt.imshow(sobel[i], 'gray'), plt.axis('off')
            except IndexError as e:
                warnings.warn(repr(e))
                continue

        return handle

    def __show_3(self):
        sobels = self.getSobelPic()
        sobels.append(self.AdaptiveThresholdPic())
        handle = plt.figure("show_3")
        for sobel, idx in zip(sobels, range(len(sobels))):
            plt.subplot(1, len(sobels) + 1, idx + 1)
            plt.imshow(self.GetShapeMask(sobel), 'gray'), plt.axis('off')
        plt.show()
        return handle


def test_Filter():
    pass


if __name__ == '__main__':
    test_Filter()
