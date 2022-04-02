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
    def FromBinPic(cls, pic, detectFun=LinePointDetectCentralize):
        if pic.ndim > 2:
            raise ValueError("the input should be a bin Pic")
        obj = detectFun(pic)
        length = len(obj)
        if length == 2:
            return cls(obj[0], obj[1])
        elif length == 4:
            return cls(obj[0], obj[1], obj[2], obj[3])
        else:
            raise ValueError("invalid func with unfitted outputs")

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

    def GetPercentageResult(self, percent=0.0) -> float:
        pos = len(self.y)
        return self.y[int(pos * percent)]

    def GetTestResult(self):
        res = []
        for i in [0, 0.25, 0.5, 0.75, 0.99]:
            res.append(self.GetPercentageResult(i))
        return res

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item: int):
        return self.x[item], self.y[item]


class FigureInfo(PointDetector):
    def __init__(self, figure):
        pos = PointDetector.FromBinPic(figure.BinPic_SmoothOutput())
        super(FigureInfo, self).__init__(pos.x, pos.y, pos.x_all, pos.y_all)
        self.figure = figure
        self.Certification()

    def Certification(self):
        shape = self.figure.BinPic_SmoothOutput().shape
        y_scale = self.figure.picLabel[0][0] / (shape[0] * 0.75)
        self.x, self.y = self.PointsTrans(int(-0.125 * shape[1]), int(-0.125 * shape[0]), scale_y= y_scale)


if __name__ == '__main__':
    pass
