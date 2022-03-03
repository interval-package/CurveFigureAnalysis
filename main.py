# This is a sample Python script.
import cv2
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    image = cv2.imread("../data/img_train_BrokenLine/%d/draw.png" % 94).astype(np.float32)
    print(np.array((104.00698793,  # Minus statistics.
                                  116.66876762,
                                  122.67891434)))
    image = image - np.array((104.00698793,  # Minus statistics.
                              116.66876762,
                              122.67891434))
    image_1 = image
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW.
    image = image.astype(np.float32)  # To float32.
    plt.imshow(image_1)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
