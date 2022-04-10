import math
from matplotlib import pyplot as plt

from Output_Action.ResultProducer import *

from DetectionWrapper.PointDetector import PointDetector

from LineClasses.BrokenLineFigure import *
from LineClasses.CurveFigure import *


def AdaptiveShow(inputs: list):
    l = len(inputs)
    n = math.ceil(math.sqrt(l))
    fig = plt.figure("compiled show")
    for i in range(l):
        plt.subplot(n, n, i + 1), plt.imshow(inputs[i], 'gray'), plt.axis('off')
    return fig


def DispPoints(x, y):
    fig = plt.figure("Points output")
    plt.plot(x, y)
    return fig


def Display_AllPoints(obj: FigureInfo):
    poi = obj.Get_Poi_Hierarchy()
    # plt.figure()
    # plt.plot(poi.x, poi.y)
    # poi.Display_All()
    poi.AlterInit_Correction()
    plt.subplot(1, 2, 1)
    plt.plot(poi.x, poi.y)
    tag = obj.figure.picLabel[1]
    tag = np.array(tag)
    print("label:", tag)
    plt.plot(poi.x_pos, tag, 'rx')

    # res = obj.Output_Central_Fit_Correction()
    # print("dif:", (tag-res)/tag)
    # plt.plot(poi.x_pos, res, 'gx')

    res = obj.Output_Central_Interp_Hierarchy()
    print("dif:", (tag - res) / tag)
    plt.plot(poi.x_pos, res, 'yx')

    # plt.plot(poi.peaks, poi.y[poi.peaks], 'x')

    plt.subplot(2, 2, 2)
    plt.imshow(obj.figure.rawPic)

    plt.subplot(2, 2, 4)
    plt.plot(poi.x_all, poi.y_all, '.')
    plt.plot(poi.x, poi.y)

    # plt.figure()
    # plt.imshow(obj.figure.rawPic)
    plt.show()


def Display_AllPoints_Test(obj: FigureInfo):
    poi = obj.Get_Poi_Hierarchy()
    poi.AlterInit_Correction()

    print("peaks+trough:count", len(poi.peaks)+len(poi.trough))

    poi.AlterInit_Correction()
    plt.subplot(1, 2, 1)
    plt.plot(poi.x, poi.y)
    res = obj.Output_Central_Interp_Hierarchy()
    print(res)
    plt.plot(poi.x_pos, res, 'yx')

    plt.subplot(2, 2, 2)
    plt.imshow(obj.figure.rawPic)

    plt.subplot(2, 2, 4)
    plt.plot(poi.x_all, poi.y_all, '.')
    plt.plot(poi.x, poi.y)

    plt.show()


def ImCorrection(obj: FigureInfo):
    poi = obj.Get_Poi_Hierarchy()
    res = obj.figure.picLabel[1]


if __name__ == '__main__':
    pass
