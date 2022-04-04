import cv2
import numpy as np
import matplotlib.pyplot as plt


def PicTrans2HEDInput(pic: np.ndarray, size=None) -> np.ndarray:
    if size is None:
        width, height = 480, 320
    else:
        width, height = size
    cols, rows, _ = pic.shape
    matrix = np.float32([[width / rows, 0, 0],
                         [0,height / cols,  0],
                         [0, 0, 1]])
    output = cv2.warpPerspective(src=pic, M=matrix, dsize=(width, height))
    return output


if __name__ == '__main__':
    pic = cv2.imread("../../../data/img_train_BrokenLine/3/draw.png")
    plt.subplot(2, 2, 1)
    plt.imshow(pic)
    plt.subplot(2, 2, 2)
    plt.imshow(PicTrans2HEDInput(pic))
    plt.show()
    pass
