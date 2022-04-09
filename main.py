import matplotlib.pyplot as plt

from utils.dispUtils import *

from Output_Action.ResultProducer import *

from DetectionWrapper.PointDetector import PointDetector

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = FigureInfo(BrokenLineFigure.fromId_TrainingSet(i))
    label = obj.figure.picLabel[1]
    tar = obj.Display_Hierarchy()
    res = tar.GetResult_TarVector_ByX(tar.GetResult_Specific_ByX_Centralized_Fitted_Insert)
    res_2 = tar.GetResult_TarVector_ByX(tar.GetResult_Specific_ByX_Centralized)
    f = tar.Display_All()
    plt.plot(tar.x, tar.y)
    plt.plot(PointDetector.x_pos, res, '.r')
    plt.plot(PointDetector.x_pos, res_2, '.y')
    plt.plot(PointDetector.x_pos, label, '.g')
    plt.show()
    pass


if __name__ == '__main__':
    obj = ResultProducer(0, 500)
    obj.ProduceToExcel()
    # for i in range(30, 50):
    #     main(i)
    pass
