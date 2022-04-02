import os

import numpy as np
import matplotlib.pyplot as plt

from utils.dispUtils import *

from DetectionWrapper.ResultProducer import *

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = BrokenLineFigure.fromId_TrainingSet(i)

    t_res = np.array(obj.picLabel[1])
    pos = FigureInfo(obj)
    res = np.array(pos.GetTestResult())

    print((t_res - res) / t_res)

    AdaptiveShow(obj.BinPic_SetGetter())
    plt.show()
    # input("stop:")
    pass


if __name__ == '__main__':
    obj = ResultProducer(0, 500)
    obj.ProduceToExcel()
    pass
