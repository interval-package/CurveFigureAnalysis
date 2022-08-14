import numpy as np

from Output_Action.FigureInfo import FigureInfo
from CurceFigureAnalysis.LineClasses.BrokenLineFigure import BrokenLineFigure
from CurceFigureAnalysis.LineClasses.CurveFigure import CurveFigure

from CurceFigureAnalysis.utils.ExceptionClasses import *

import xlwt


class ResultProducer(object):
    def __init__(self, start, end, Path_DataSet=None, Tar_Type=0):
        if Path_DataSet is None:
            if Tar_Type == 0:
                self.path = '../data/img_test_BrokenLine/'
            else:
                self.path = '../data/img_test_Curve/'
        else:
            self.path = Path_DataSet

        self.start = start
        self.end = end

        self.type = 0

        pass

    def SetTestSet(self, Tar_Type=0):
        self.type = 1
        if Tar_Type == 0:
            self.path = '../data/img_test_BrokenLine/'
        else:
            self.path = '../data/img_test_Curve/'

    def __getitem__(self, index):
        if self.type == 1:
            obj = FigureInfo(CurveFigure.fromId_TestingSet(index)).Get_Poi_Hierarchy()
            obj.AlterInit_Correction()
        else:
            obj = FigureInfo(BrokenLineFigure.fromId_TestingSet(index)).Get_Poi_Hierarchy()
            obj.AlterInit_Correction()
        res = np.ones((1, 5))
        try:
            res = obj.GetResult_TarVector_ByX(obj.GetResult_Specific_ByX_Centralized)
        except OutputErrorOfBadQuality as e:
            print(repr(e))
            pass
        return res

    def ProduceToSheet(self, workbook, sheet_name='Sheet 1'):
        sheet = workbook.add_sheet(sheet_name)
        for row in range(self.start, self.end):
            res = self[row]
            print(row)
            for col in range(5):
                sheet.write(row, col, res[col])
        pass

    def ProduceToExcel(self, SavePath='./Output.xls'):
        workbook = xlwt.Workbook()
        self.type = 1
        self.ProduceToSheet(workbook, 'sheet 1')
        self.type = 0
        self.ProduceToSheet(workbook, 'sheet 2')
        workbook.save(SavePath)

    def __len__(self):
        return self.end - self.start + 1


if __name__ == '__main__':
    path = '../../data/img_train_BrokenLine/'
    pass
