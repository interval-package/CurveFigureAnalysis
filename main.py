import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = BrokenLineFigure.fromFile(i)
    tar = obj.picLabel
    t_res = np.array(tar[1])
    print(t_res)
    pos = FigureInfo(obj)
    try:
        res = np.array(pos.GetTestResult())
        print(res)
        print((t_res-res)/t_res)
    except Exception as e:
        print(repr(e))
        pass
    plt.subplot(2, 2, 1)
    plt.imshow(obj.rawPic)
    plt.subplot(2, 2, 2)
    plt.plot(pos.x, pos.y)
    plt.subplot(2, 2, 3)
    plt.imshow(obj.BinPic_SmoothOutput(), 'gray')
    plt.show()

    # input("stop:")
    pass


if __name__ == '__main__':
    for i in range(20, 100):
        main(i)
    pass
