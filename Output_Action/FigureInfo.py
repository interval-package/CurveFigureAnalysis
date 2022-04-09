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

    def Output_GroupDecision_Vote(self):
        tar = []
        for i in self.Pos_Set:
            tar.append(i.GetResult_TarVector_ByX(i.GetResult_Specific_ByX_Centralized_Fitted_Insert))
        return tar

    def Output_GroupDecision_Hierarchy(self):
        for i in self.Pos_Set:
            tar = i.GetResult_TarVector_ByX(i.GetResult_Specific_ByX_Centralized_Fitted_Insert)
            if tar is not None:
                return tar
        pass

    def Output_Central_Fit_Hierarchy(self):
        for pos in self.Pos_Set:
            tar = pos.GetResult_TarVector_ByX(pos.GetResult_Specific_ByX_Centralized)
            if tar is not None:
                return tar
        return

    def Output_Mean_Hierarchy(self):
        for pos in self.Pos_Set:
            tar_1 = pos.GetResult_TarVector_ByX(pos.GetResult_Specific_ByX_Centralized)
            tar_2 = pos.GetResult_TarVector_ByX(pos.GetResult_Specific_ByX_Centralized_Fitted_Insert)
            if tar_1 is not None and tar_2 is not None:
                tar = (tar_1+tar_2)
                for i in range(5):
                    tar[i] /= 2
                return tar
        return

    def Get_Poi_Hierarchy(self):
        for pos in self.Pos_Set:
            tar = pos.GetResult_TarVector_ByX(pos.GetResult_Specific_ByX_Centralized_Fitted_Insert)
            if tar is not None:
                return pos
        pass

    def Output_raw(self):
        raise OutputErrorOfBadQuality()

    def display(self, func_res):
        f = plt.figure()
        plt.plot(self.figure.picLabel[1])
        for poi in self.Output_GroupDecision_Vote():
            if poi is None:
                continue
            plt.plot(poi, '.')
        return f

    @staticmethod
    def Output_Bad_Result():
        return [0, 0, 0, 0, 0]
