from DetectionWrapper.PointDetectFunctions import *


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

    def PointsTrans(self, x=0, y=0, scale_x=1, scale_y=1):
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
        x = (self.x + x) * scale_x
        y = (self.y + y) * scale_y
        return x, y

    def PointsTrans_Targeted(self, shape, max_val):
        y_scale = max_val / (shape[0] * 0.75)
        self.x, self.y = self.PointsTrans(int(-0.125 * shape[1]),
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

    def GetResult_Specific_ByX_Clump(self, pos, Deciding_Method, window=5):
        tar_y = Deciding_Method(self.y_all[self.x_all == pos])
        if tar_y is None:
            tar_y = self.y[self.x == pos]
        return tar_y

    def GetResult_Specific_ByX_PeakDecide(self):
        return

    x_len = 360

    def GetResult_TarVector_ByX(self, func):
        res = []
        for i in [0, 0.25, 0.5, 0.75, 0.99]:
            res.append(func(int(i * self.x_len)))
        return res


if __name__ == '__main__':
    pass
