from DetectionWrapper.PointDetector import *
from LineClasses.LineFigure import LineFigure

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
        pass

    def SetTestSet(self, Tar_Type=0):
        if Tar_Type == 0:
            self.path = '../data/img_test_BrokenLine/'
        else:
            self.path = '../data/img_test_Curve/'

    def __getitem__(self, index):
        res = [0, 0, 0, 0, 0]
        try:
            res = FigureInfo(LineFigure.fromFile(self.path + '{}'.format(index))).GetResult()
        except IndexError as e:
            print(repr(e))
        except ValueError as e:
            print(repr(e))
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
        self.SetTestSet(1)
        self.ProduceToSheet(workbook, 'sheet 1')
        self.SetTestSet(0)
        self.ProduceToSheet(workbook, 'sheet 2')
        workbook.save(SavePath)

    def __len__(self):
        return self.end - self.start + 1


if __name__ == '__main__':
    path = '../../data/img_train_BrokenLine/'
    pass
