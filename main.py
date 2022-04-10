from init import *

import matplotlib.pyplot as plt

from utils.dispUtils import *



UsingHed = False


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

    bad_set = [7, 12, 40, 47, 53, 71, 86, 101, 130, 145, 146]

    for i in bad_set:
        main(i)

    for i in range(0, 500):
        main(i)
    pass
