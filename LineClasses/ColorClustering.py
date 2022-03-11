import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def movePic(pic, x=0, y=0):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols = pic.shape[0:2]
    res = cv2.warpAffine(pic, mat, (cols, rows))
    return res


def GetColorBoundary(pic):
    """
    @:param pic:should be the BGR version(the opencv default), we will change it into HSv
    """
    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)

    def picFilter(h_l, h_h, s_l, s_h, v_l, v_h):
        res = cv2.inRange(pic, (h_l, s_l, v_l), (h_h, s_h, v_h))
        return res

    figrue = plt.imshow(pic)

    axamp = plt.axes([0.1, 0.25, 0.0225, 0.63])
    amp_slider = Slider(
        ax=axamp,
        label="Amplitude",
        valmin=0,
        valmax=255,
        valinit=init_amplitude,
        orientation="vertical"
    )

class ClusteringNode(object):
    def __init__(self, pic: np.ndarray, mask, objType="training"):
        """
        @:param pic:should be the BGR version(the opencv default), we will change it into HSv
        @:param objType: should be training or test, but actually if not training we all recognize as the test
        """
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        self.vector = cv2.calcHist(pic, [0, 1, 2], self.getSimpleMask(pic), [8, 8, 8],
                                   [0, 256, 0, 256, 0, 256]).flatten()
        if objType != 'training':
            self.tar = cv2.calcHist(pic, [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten()

    @staticmethod
    def getSimpleMask(pic):
        """
        :return 通过设定的预期值获取最基本的蒙版
        """
        rows, cols = pic.shape[:2]
        maskArea = np.zeros([rows, cols], dtype=np.uint8)
        maskArea[int(rows * 0.125):int(rows * 0.875), int(cols * 0.127):int(cols * 0.9)] = 255
        return maskArea

    def __getitem__(self, item):
        return self.vector[item]

    def __len__(self):
        return self.vector.shape[1]


if __name__ == '__main__':
    pass
