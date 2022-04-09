import matplotlib.pyplot as plt
import numpy as np

from DetectionWrapper.PointDetectFunctions import *
from DetectionWrapper.PointDecide_Methods import *
from utils.ExceptionClasses import *
from utils.picProcessors import IsBinPicValid


class PointDetector(object):
    def __init__(self, x, y, x_all=None, y_all=None):
        self.x, self.y = x, y
        if x_all is None or y_all is None:
            self.x_all = x
            self.y_all = y
        else:
            self.x_all = x_all
            self.y_all = y_all
        pass

    @classmethod
    def FromBinPic(cls, pic, max_val=None, detectFun=LinePointDetectCentralize):
        # if ~IsBinPicValid(pic):
        #     raise OutputErrorOfBadQuality()
        if pic.ndim > 2:
            raise ValueError("the input should be a bin Pic")
        obj = detectFun(pic)
        length = len(obj)
        if length == 2:
            tar = cls(obj[0], obj[1])
        elif length == 4:
            tar = cls(obj[0], obj[1], obj[2], obj[3])
        else:
            raise ValueError("invalid func with unfitted outputs")
        if max_val is not None:
            tar.PointsTrans_Targeted(pic.shape, max_val)
            tar.MissedPoints_Fill_Interp()
        return tar

    def PointsTrans(self, x=0, y=0, scale_x=1, scale_y=1, ):
        """
        parameters
        ==============
        :param x: move the points on the x-axis
        :param y: move the points on the y-axis
        :param scale_x:
        :param scale_y:
        :returns
        ==============
        :return: this function will change the points within, and return the changed
        """
        self.x = (self.x + x) * scale_x
        self.y = (self.y + y) * scale_y

        self.x_all = (self.x_all + x) * scale_x
        self.y_all = (self.y_all + y) * scale_y
        return

    # we assume that after trans the end of line are at the pos x = x_len
    x_len = 370

    x_pos = [int(i * 370) for i in [0, 0.25, 0.5, 0.75, 1]]

    def PointsTrans_Targeted(self, shape, max_val):
        y_scale = max_val / (shape[0] * 0.75)
        # print(int(-0.125 * shape[1]), int(-0.125 * shape[0]), (shape[0] * 0.75), max_val)
        self.PointsTrans(int(-0.13 * shape[1]),
                         int(-0.125 * shape[0]), scale_y=y_scale)

    def MissedPoints_Fill_Interp(self):
        x = np.arange(0, self.x_len + 1)
        try:
            self.y = np.interp(x, self.x, self.y)
            self.x = x
        except ValueError as e:
            pass
        pass

    def Points_To_Peaks(self):
        return

    # output methods
    # if Specific returns x
    # if var tar returns a vector of 5 at targeted pos

    # percentage methods

    def GetResult_Specific_ByPercentage(self, percent=0.0) -> float:
        pos = len(self.y)
        try:
            tar = self.y[int(pos * percent)]
        except IndexError as e:
            raise OutputErrorOfBlank(repr(e))
        return tar

    def GetResult_TarVector_ByPercentage(self):
        res = []
        try:
            for i in [0, 0.25, 0.5, 0.75, 0.99]:
                res.append(self.GetResult_Specific_ByPercentage(i))
        except OutputErrorOfBlank as e:
            print(repr(e))
            return None
        return res

    # by x methods

    def GetResult_Specific_ByX_Centralized(self, pos):
        tar = self.y[self.x == pos]
        if len(tar) == 0:
            raise OutputErrorOfSpecificPos("point do not found")
        return tar[0]

    def GetResult_Specific_ByX_Centralized_Fitted_Insert(self, pos, window=10):
        tar = np.bitwise_and(self.x_all > pos - window, self.x_all < pos + window)
        x = self.x_all[tar]
        y = self.y_all[tar]
        if len(x) < 3 or len(y) < 3:
            if window > 6:
                raise OutputErrorOfBlank("")
            else:
                raise OutputErrorOfSpecificPos("")
        try:
            p1 = np.poly1d(np.polyfit(x, y, 3))
        except Exception as e:
            raise OutputErrorOfSpecificPos(repr(e))
        return p1(pos)

    def GetResult_Specific_ByX_Centralized_Interp_Insert(self, pos, window=10):
        tar = np.bitwise_and(self.x > pos - window, self.x < pos + window)
        x = self.x[tar]
        y = self.y[tar]
        if len(x) == 0 or len(y) == 0:
            if window > 6:
                raise OutputErrorOfBlank("")
            else:
                raise OutputErrorOfSpecificPos("")
        return np.interp(pos, x, y)[0]

    def GetResult_Specific_ByX_Clump(self, pos, deciding_method=DoNothing, window=5):
        try:
            tar_y = deciding_method(self.GetSlice_ByX(pos, window))
        except OutputErrorOfSpecificPos as e:
            return None
        return tar_y

    def GetResult_Specific_ByX_PeakDecide(self, pos):
        return

    def GetResult_TarVector_ByX(self, func=None):
        if func is None:
            func = self.GetResult_Specific_ByX_Centralized
        res = []
        for i in [0, 0.25, 0.5, 0.75, 1]:
            try:
                res.append(func(int(i * self.x_len)))
            except IndexError as e:
                res.append(None)
            except OutputErrorOfSpecificPos:
                res.append(None)
            except OutputErrorOfBlank:
                return None
        return res

    # get sliced method

    def GetSlice_ByX(self, pos, window=5):
        pos_bin = np.bitwise_and(self.x_all > pos - window, self.x_all < pos + window)
        if len(pos_bin) == 0:
            return None
        tar_y = self.y_all[pos_bin]
        tar_x = self.x_all[pos_bin]
        return tar_x, tar_y

    # display methods

    def Display_Sliced(self):
        tar = self.GetResult_TarVector_ByX(func=self.GetSlice_ByX)
        size = len(tar)
        f = plt.figure()
        try:
            for item, i in zip(tar, range(size)):
                if item is None:
                    continue
                plt.subplot(2, 3, i + 1)
                plt.plot(item[0], item[1], '.')
        except ValueError as e:
            # print(repr(e))
            return None
        return f

    def Display_All(self, sliced_info=False):
        f = plt.figure()
        plt.plot(self.x_all, self.y_all, '.')
        if sliced_info:
            tar = self.GetResult_TarVector_ByX(func=self.GetSlice_ByX)
            for item in tar:
                if item is None:
                    continue
                plt.plot(item[0], item[1], '.', color='red')
        return f


if __name__ == '__main__':
    pass
