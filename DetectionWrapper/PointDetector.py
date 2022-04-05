import matplotlib.pyplot as plt

from DetectionWrapper.PointDetectFunctions import *
from DetectionWrapper.PointDecide_Methods import *


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

    def PointsTrans_Targeted(self, shape, max_val):
        y_scale = max_val / (shape[0] * 0.75)
        # print(int(-0.125 * shape[1]), int(-0.125 * shape[0]), (shape[0] * 0.75), max_val)
        self.PointsTrans(int(-0.125 * shape[1]),
                         int(-0.125 * shape[0]), scale_y=y_scale)

    def GetResult_Specific_ByPercentage(self, percent=0.0) -> float:
        pos = len(self.y)
        return self.y[int(pos * percent)]

    def GetResult_TarVector_ByPercentage(self):
        res = []
        try:
            for i in [0, 0.25, 0.5, 0.75, 0.99]:
                res.append(self.GetResult_Specific_ByPercentage(i))
        except IndexError as e:
            print(repr(e))
            res = [0, 0, 0, 0, 0]
        return res

    def GetResult_Specific_ByX_Centralized(self, pos):
        tar = self.y[self.x == pos]
        if len(tar) == 0:
            raise IndexError("untracked error")
        return tar[0]

    def GetResult_Specific_ByX_Centralized_Inserted(self, pos):
        return

    def GetResult_Specific_ByX_Clump(self, pos, Deciding_Method=DoNothing, window=5):
        tar_y = Deciding_Method(self.y_all[self.x_all == pos])
        if tar_y is None:
            tar_y = self.y[self.x == pos]
        return tar_y

    def GetResult_Specific_ByX_PeakDecide(self):
        return

    x_len = 380

    def GetResult_TarVector_ByX(self, func=None):
        if func is None:
            func = self.GetResult_Specific_ByX_Clump
        res = []
        for i in [0, 0.25, 0.5, 0.75, 0.99]:
            try:
                res.append(func(int(i * self.x_len)))
            except IndexError as e:
                print(repr(e))
                res.append(0)
        return res

    def GetSlice_ByX(self, pos, window=5):
        pos_bin = np.bitwise_and(self.x_all > pos - window, self.x_all < pos + window)
        if len(pos_bin) == 0:
            return None
        tar_y = self.y_all[pos_bin]
        tar_x = self.x_all[pos_bin]
        return tar_x, tar_y

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
