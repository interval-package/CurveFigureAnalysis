from utils.dispUtils import *

from Output_Action.ResultProducer import *

from LineClasses.BrokenLineFigure import *


def main(i):
    obj = FigureInfo(BrokenLineFigure.fromId_TrainingSet(i))
    print("liable:", obj.figure.picLabel)
    for i in obj.Pos_Set:
        # i.Display_All(True)
        print(i.GetResult_TarVector_ByX(func=i.GetResult_Specific_ByX_Centralized))
        print()
        # i.Display_Sliced()
    # f = plt.figure()
    # plt.imshow(obj.figure.rawPic)
    # plt.show()
    print("\ngap\n")
    pass


if __name__ == '__main__':
    # obj = ResultProducer(0, 500)
    # obj.ProduceToExcel()
    for i in range(200):
        main(i)
    pass
