import cv2
import numpy as np


def PicTrans2HEDInput(pic: np.ndarray, size=None) -> np.ndarray:
    if size is None:
        width, height = 480, 320
    else:
        width, height = size
    rows, cols, _ = pic.shape
    matrix = np.float32([[width / rows, 0, 0],
                         [0, height / cols, 0],
                         [0, 0, 1]])
    output = cv2.warpPerspective(pic, matrix, (width, height, 3))
    return output


if __name__ == '__main__':
    pass
