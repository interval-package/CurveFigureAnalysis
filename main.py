import matplotlib.pyplot as plt

from utils.dispUtils import *

from Output_Action.ResultProducer import *

from DetectionWrapper.PointDetector import PointDetector

from LineClasses.BrokenLineFigure import *


def Display_allRawPic(in_: FigureInfo):
    obj = in_.figure
    group = obj.BinPic_SetGetter()
    plt.subplot(2, 2, 1)
    plt.imshow(group[0], 'gray')
    plt.subplot(2, 2, 2)
    plt.imshow(np.bitwise_xor(group[0], obj.givenPic), 'gray')
    plt.subplot(2, 2, 3)
    plt.imshow(obj.rawPic)
    plt.subplot(2, 2, 4)
    plt.imshow(obj.givenPic, 'gray')
    plt.show()


def Display_AllPoints(obj: FigureInfo):
    poi = obj.Get_Poi_Hierarchy()
    poi.Display_All()
    plt.plot(poi.x, poi.y)
    plt.plot(poi.x_pos, obj.figure.picLabel[1])
    plt.figure()
    plt.imshow(obj.figure.rawPic)
    plt.show()


def main(i):
    obj = FigureInfo(BrokenLineFigure.fromId_TrainingSet(i))
    Display_AllPoints(obj)
    pass


if __name__ == '__main__':
    # obj = ResultProducer(0, 500)
    # obj.ProduceToExcel()
    for i in range(30, 50):
        main(i)
    pass
