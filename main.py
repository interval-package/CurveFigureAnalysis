import cv2
import matplotlib.pyplot as plt
import numpy as np
from LineClasses.LineFigure import LineFigure


def main(id:int):
    LineFigure.fromFile("../data/img_train_BrokenLine/%d" % id,True)
    pass


if __name__ == '__main__':
    main(15)
    pass
