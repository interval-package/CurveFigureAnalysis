import matplotlib.pyplot as plt

from utils.dispUtils import *

from Output_Action.ResultProducer import *

from DetectionWrapper.PointDetector import PointDetector

from LineClasses.BrokenLineFigure import *
from LineClasses.CurveFigure import *


def Pic_Quality():
    return


def Display_allRawPic(in_: FigureInfo):
    obj = in_.figure
    group = obj.BinPic_SetGetter()
    plt.subplot(2, 2, 1)
    plt.imshow(group[0], 'gray')
    try:
        plt.subplot(2, 2, 2)
        plt.imshow(group[1], 'gray')
    except IndexError:
        pass
    try:
        plt.subplot(2, 2, 3)
        plt.imshow(group[2], 'gray')
    except IndexError:
        pass
    try:
        plt.subplot(2, 2, 4)
        plt.imshow(group[3], 'gray')
    except IndexError:
        pass
    plt.show()


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


def main_Output():
    obj = ResultProducer(0, 500)
    obj.ProduceToExcel()


def main(i):
    # obj = FigureInfo(BrokenLineFigure.fromId_TrainingSet(i))

    # obj = FigureInfo(CurveFigure.fromId_TrainingSet(i))
    # Display_AllPoints(obj)
    print(i)
    obj = FigureInfo(BrokenLineFigure.fromId_TestingSet(i))
    Display_allRawPic(obj)
    Display_AllPoints_Test(obj)
    pass


if __name__ == '__main__':

    # main_Output()
    # main(40)

    bad_set = [7, 12, 40, 47, 53, 71, 86, 101, 130, 145, 146]

    for i in bad_set:
        main(i)

    for i in range(0, 500):
        main(i)
    pass
