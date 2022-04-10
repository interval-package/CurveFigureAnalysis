import cv2
import numpy as np
import pandas as pd


# utils
def IsBinPicValid(pic: np.ndarray, gap=10000) -> bool:
    """
    :param pic: binary pic
    :param gap: the minimal num of valid points
    :return: bool, the pic is valid or not
    """
    # 图像有效性验证，如果图像内部为真的像素点过少，则验证失败
    if pic.ndim > 2:
        raise ValueError("you should input a binary pic for certification")
    hist_inner = cv2.calcHist([pic], [0], None, [2], [0, 256])
    # print(hist_inner)
    return hist_inner[-1] > gap


# 读取文件
def readPicFromFile(basicPath: str):
    # 标准化读取图片信息
    # print(basicPath + "/draw.png")
    rawPic = cv2.imread(basicPath + "/draw.png")
    if rawPic is None:
        raise ValueError("invalid path")
    binary_pic = None
    try:
        with open(basicPath + "/db.txt") as f:
            # w, h = [int(x) for x in next(f).split()]
            pic_label = [[float(x) for x in line.split(',')] for line in f]
    except IOError:
        pic_label = [100]

        binary_pic = None

    if len(pic_label) > 1:
        try:
            binary_pic = cv2.imread(basicPath + "/draw_mask.png")
            # pic_label = pd.read_table(basicPath + "/db.txt", engine='python', delimiter="\n")
        except IOError as IOe:
            print('repr(IOe):\t', repr(IOe))
        except Exception as e:
            print('repr(e):\t', repr(e))

    return rawPic, binary_pic, pic_label


def eraseFrame(img: np.ndarray) -> np.ndarray:
    if img.ndim > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    ret, binary = cv2.threshold(gray, 200, 255, 0)
    # assuming, b_w is the binary image
    inv = 255 - binary
    horizontal_img = inv
    vertical_img = inv

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
    horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
    vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

    mask_img = horizontal_img + vertical_img
    # no_border = np.bitwise_or(binary, mask_img)
    no_border = cv2.bitwise_or(binary, mask_img)

    return no_border
