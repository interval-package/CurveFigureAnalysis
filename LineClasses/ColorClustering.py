import cv2
import numpy as np
import matplotlib.pyplot as plt


def GetMaskSimple(rawPic):
    rows, cols = rawPic.shape[:2]
    maskArea = np.zeros([rows, cols], dtype=np.uint8)
    # x, y, w, h = self.box.rectBox[1:]
    # maskArea[int(y):int(y+h),int(x):int(x+w)] = 255
    maskArea[int(rows / 7):int(7 * rows / 8), int(cols / 8) + 1:int(8 * cols / 9) + 7] = 255
    return maskArea


def movePic(pic, x=0, y=0):
    mat = np.float32([[1, 0, x], [0, 1, y]])
    rows, cols = pic.shape[0:2]
    res = cv2.warpAffine(pic, mat, (cols, rows))
    return res


class ColorCluster(object):
    def __init__(self, pic: np.ndarray, mask=None):
        self.pic = pic
        if mask is None:
            self.mask = GetMaskSimple(self.pic)
        else:
            self.mask = mask
        pass

    def CalcSelf(self):
        gray = cv2.cvtColor(self.pic, cv2.COLOR_BGR2GRAY)
        hist_0 = cv2.calcHist([self.pic], [0], self.mask, [256], [0, 255]).reshape(256)  # open distinct
        hist_1 = cv2.calcHist([self.pic], [1], self.mask, [256], [0, 255]).reshape(256)
        hist_2 = cv2.calcHist([self.pic], [2], self.mask, [256], [0, 255]).reshape(256)
        hist_gray = cv2.calcHist([gray], [0], self.mask, [256], [0, 255]).reshape(256)
        x = np.arange(0, 256)
        return hist_0, hist_1, hist_2, hist_gray, x

    def show(self):
        fig = plt.figure()
        hist_0, hist_1, hist_2, hist_gray, x = self.CalcSelf()
        plt.subplot(2, 2, 1), plt.bar(x=x[hist_0 > 0], height=hist_0[hist_0 > 0], color="b"), plt.title(
            "color hist b")
        plt.subplot(2, 2, 2), plt.bar(x[hist_1 > 0], hist_1[hist_1 > 0], color="g"), plt.title(
            "color hist g")
        plt.subplot(2, 2, 3), plt.bar(x[hist_2 > 0], hist_2[hist_2 > 0], color="r"), plt.title(
            "color hist r")
        plt.subplot(2, 2, 4), plt.bar(x[hist_gray > 0], hist_gray[hist_gray > 0]), plt.title(
            "color hist gray")
        return fig


def main(id):
    src = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
    rows, cols, canals = src.shape
    maskpic_target = cv2.imread("../../data/img_train_BrokenLine/%d/draw_mask.png" % id)
    maskpic_target = cv2.cvtColor(maskpic_target, cv2.COLOR_BGR2GRAY)
    maskpic_target = cv2.threshold(maskpic_target, 10, 255, cv2.THRESH_BINARY, 1)[1]
    # maskpic_target[int(0.4 * rows):int(0.6 * rows), int(0.4 * cols):int(0.6 * cols)] = 255
    maskpic_target = movePic(maskpic_target, -3, -1)

    ColorCluster(src).show()
    ColorCluster(src, maskpic_target).show()
    plt.show()
    # plt.figure()
    # plt.subplot(1, 2, 1), plt.imshow(src)
    # plt.subplot(1, 2, 2), plt.imshow(cv2.bitwise_and(src, src, mask=maskpic_target))
    # plt.show()


if __name__ == '__main__':
    for i in range(1, 100):
        main(i)
    pass
