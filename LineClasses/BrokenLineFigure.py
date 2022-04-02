from functools import singledispatch
import matplotlib.pyplot as plt
from LineClasses.LineFigure import *
import numpy as np
import cv2


class BrokenLineFigure(LineFigure):
    @singledispatch
    def __init__(self, rawPic, givenPic=None, picLabel=None):
        # read in the pic into the obj
        super().__init__(rawPic, givenPic, picLabel)
        # super(BrokenLineFigure, self).__init__("../../data/img_train_BrokenLine/%d" % id, test)
        self.type = "BrokenLine"
        # 待定，可能使用PointDetector作为基类
        self.pointsObj = PointDetector.FromBinPic(self.BinPic_SmoothOutput())
        # 最好的目标是，折线图最后提取出来是一组点坐标

    @classmethod
    def fromFile(cls, id: int):
        rawPic, givenPic, picLabel = readPicFromFile("../data/img_train_BrokenLine/%d" % id)
        return cls(PicTrans2HEDInput(rawPic), givenPic, picLabel)

    def TurningPointGet(self):
        if self.processedPic is not None:
            pic = self.processedPic
        else:
            pic = self.BinPic_SmoothOutput()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pic = cv2.morphologyEx(pic, cv2.MORPH_BLACKHAT, kernel)
        return pic

    def getPoints(self):
        if self.processedPic is not None:
            pic = self.processedPic
        else:
            pic = self.BinPic_SmoothOutput()
        # 中心法获取图像点
        x, y, x_c, y_c = LinePointDetectCentralize(pic)



if __name__ == '__main__':
    for i in range(100, 230):
        obj = BrokenLineFigure.fromFile(i)
        obj.getPoints()
    pass
