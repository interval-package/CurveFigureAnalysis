import cv2
import numpy as np
import matplotlib.pyplot as plt


# 三通道分割
# path_0, path_1, path_2 = spliter(result)
# images = [result, path_0, path_1, path_2]
# path_0, path_1, path_2 = cv2.split(result)
def spliter(rawpic):
    path_0 = rawpic[:, :, 0]
    path_1 = rawpic[:, :, 1]
    path_2 = rawpic[:, :, 2]
    return path_0, path_1, path_2


# 外包装SideFilter
def FilterSelf(self):
    if self.testVersion:
        img = self.SideFilter(self.processedPic, self.GetColorBoundry()[1])
    else:
        img = self.SideFilter(self.rawPic, self.GetColorBoundry()[1])
    if ~(self.binaryPic.any() == np.ndarray([])):
        self.binaryPic = img
    return img


def show(images):
    # 多图显示
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1), plt.imshow(images[i], cmap=plt.cm.gray)
        # 取消坐标显示
        # plt.xticks([]), plt.yticks([])
        plt.axis('off')
    plt.show()


images = []
id = 0
rawpic = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
maskpic = cv2.imread("../../data/img_train_BrokenLine/%d/draw.png" % id)
# 实现BGR到RGB转变
result = cv2.cvtColor(rawpic, cv2.COLOR_RGB2GRAY)

rows, cols = result.shape[:2]
images.append(result)
# r, g, b = cv2.split(rawpic)
tres, source = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
images.append(source)
images.append(cv2.GaussianBlur(source, (5, 5), 0))
images.append(cv2.medianBlur(source, 13))
# threshold type THRESH_"Name"

# 形态学处理
# cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)


# 进行边缘检测
img = rawpic
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X

tres, img_new = cv2.threshold(sobelx, 100, 255, cv2.THRESH_BINARY_INV)

# ...pixel neighborhood that is used during filtering.
# sigmaColor is used to filter sigma in the color space.
# sigmaSpace is used to filter sigma in the coordinate space.
bilateral_filter = cv2.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)

sobel = [sobelx, sobely, sobelxy, img_new, bilateral_filter]

show(sobel)
