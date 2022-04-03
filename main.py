import os

import numpy as np
import matplotlib.pyplot as plt

from utils.dispUtils import *

from DetectionWrapper.ResultProducer import *

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = BrokenLineFigure.fromId_TrainingSet(i)

    # t_res = np.array(obj.picLabel[1])
    #
    # pos = FigureInfo(obj)
    # res = np.array(pos.GetTestResult())
    #
    # print((t_res - res) / t_res)

    # pic_set = obj.BinPic_SetGetter()
    # pic_set.append(obj.BinPic_SmoothOutput())
    # pic_set.append(obj.rawPic)
    # AdaptiveShow(pic_set)
    # plt.show()
    print(obj.rawPic.shape)
    pass


if __name__ == '__main__':
    # obj = ResultProducer(0, 500)
    # obj.ProduceToExcel()
    for i in range(200):
        main(i)
    pass
