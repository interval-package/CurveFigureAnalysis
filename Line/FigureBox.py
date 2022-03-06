import cv2
import math
import matplotlib.pyplot as plt
import numpy as np


class FigureBox(object):
    def __init__(self, obj):
        self.contours = self.getContours(obj.rawPic)
        self.totalW, self.totalH = obj.rawPic.shape[:2]
        self.rectBox = self.getRectBox(self.contours)[0]
        pass

    def disp(self):
        print("Rect Box:")
        print(self.rectBox)
        return None

    def show(self):
        plt.figure("Figure Box")
        return None

    @staticmethod
    def getRectBox(contours):
        result = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # 使用边缘面积过滤较小边缘框
                [x, y, w, h] = cv2.boundingRect(cnt)
                s = w * h
                # 判断正方形拟合程度
                if math.log(abs(s - area)) < math.log(area):
                    result.append([area, x, y, w, h])
        result = np.array(result)
        return result

    @staticmethod
    def getContours(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,
                                       2)  # 自适应阈值化
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def eraseFrame(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
