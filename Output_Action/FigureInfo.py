import numpy as np

from CurceFigureAnalysis.DetectionWrapper.PointDetector import PointDetector
from CurceFigureAnalysis.LineClasses.LineFigure import LineFigure


class FigureInfo(object):
    def __init__(self, figure: LineFigure):
        self.figure = figure

        self.poi = self.Get_Poi_Hierarchy()

    # obj return

    def Get_Poi_Hierarchy(self):
        for pic in self.figure.BinPic_SetGetter():
            pos = PointDetector.FromBinPic(pic, self.figure.picLabel[0][0])
            tar = pos.GetResult_TarVector_ByX(pos.GetResult_Specific_ByX_Centralized)
            if tar is not None:
                return pos
        pass

    # Output funcs phase 1

    def Output_Central_Interp_Hierarchy(self):
        tar = self.poi.GetResult_TarVector_ByX(self.poi.GetResult_Specific_ByX_Centralized)
        if tar is not None:
            return np.array(tar)
        return

    def Output_Mean_Hierarchy(self):
        tar_1 = self.poi.GetResult_TarVector_ByX(self.poi.GetResult_Specific_ByX_Centralized)
        tar_2 = self.poi.GetResult_TarVector_ByX(self.poi.GetResult_Specific_ByX_Centralized_Fitted_Insert)
        if tar_1 is not None and tar_2 is not None:
            tar = (tar_1+tar_2)
            for i in range(5):
                tar[i] /= 2
            return tar
        raise Exception("mean fall")

    def Output_Central_Fit_Correction(self):
        tar = self.poi.GetResult_TarVector_ByX(self.poi.GetResult_Specific_ByX_Centralized_Fitted_Insert)
        if tar is not None:
            return np.array(tar)
        return None

    # Output func phase 2

    def Output_Multi_func(self):
        pass

    # display

    @staticmethod
    def Output_Bad_Result():
        return [0, 0, 0, 0, 0]
