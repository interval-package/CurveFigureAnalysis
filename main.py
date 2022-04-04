from utils.dispUtils import *

from Output_Action.ResultProducer import *

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = FigureInfo(BrokenLineFigure.fromId_TrainingSet(i))
    obj.DisplayResult()
    AdaptiveShow(obj.figure.BinPic_SetGetter())
    plt.show()
    print("\ngap\n")
    pass


if __name__ == '__main__':
    # obj = ResultProducer(0, 500)
    # obj.ProduceToExcel()
    for i in range(200):
        main(i)
    pass
