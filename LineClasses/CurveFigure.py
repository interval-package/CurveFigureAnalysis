from functools import singledispatch

from LineClasses.LineFigure import *
import cv2


class CurveFigure(LineFigure):
    @singledispatch
    def __init__(self, rawPic, givenPic=None, picLabel=None):
        # read in the pic into the obj
        super().__init__(rawPic, givenPic, picLabel)
        # super(BrokenLineFigure, self).__init__("../../data/img_train_BrokenLine/%d" % id, test)
        self.type = "BrokenLine"

    @classmethod
    def fromId_TrainingSet(cls, id: int):
        rawPic, givenPic, picLabel = readPicFromFile("../data/img_train_Curve/%d" % id)
        return cls(rawPic, givenPic, picLabel)

    @classmethod
    def fromId_TestingSet(cls, id: int):
        rawPic, givenPic, picLabel = readPicFromFile("../data/img_test_Curve/%d" % id)
        return cls(rawPic, givenPic, picLabel)

    def TurningPointGet(self):
        if self.processedPic is not None:
            pic = self.processedPic
        else:
            pic = self.BinPic_SmoothOutput()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pic = cv2.morphologyEx(pic, cv2.MORPH_BLACKHAT, kernel)
        return pic


if __name__ == '__main__':
    pass
