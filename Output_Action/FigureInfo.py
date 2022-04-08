import matplotlib.pyplot as plt

from DetectionWrapper.PointDetector import PointDetector
from LineClasses.LineFigure import LineFigure

from utils.ExceptionClasses import *


class FigureInfo(object):
    def __init__(self, figure: LineFigure):
        self.figure = figure
        self.Pos_Set = []
        for i in figure.BinPic_SetGetter():
            self.Pos_Set.append(PointDetector.FromBinPic(i, figure.picLabel[0][0]))

    def DisplayResult(self):
        for res in self.Pos_Set:
            print(res.GetResult_TarVector_ByX())

    def Output_GroupDecision_Vote(self, func_res):
        tar = []
        for i in self.Pos_Set:
            tar.append(i.GetResult_TarVector_ByX(func_res))
        
        return

    def Output_raw(self):
        raise OutputErrorOfBadQuality()

    @staticmethod
    def Output_Bad_Result():
        return [0, 0, 0, 0, 0]
