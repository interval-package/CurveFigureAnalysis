import matplotlib.pyplot as plt

from DetectionWrapper.PointDetector import PointDetector
from LineClasses.LineFigure import LineFigure


class FigureInfo(object):
    def __init__(self, figure: LineFigure):
        self.figure = figure
        self.Pos_Set = []
        for i in figure.BinPic_SetGetter():
            self.Pos_Set.append(PointDetector.FromBinPic(i, figure.picLabel[0][0]))

    def DisplayResult(self):
        for res in self.Pos_Set:
            print(res.GetResult_TarVector_ByX())

    def GetResult(self):
        res = [0, 0, 0, 0, 0]
        return res
