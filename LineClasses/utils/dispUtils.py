import math
from matplotlib import pyplot as plt


def AdaptiveShow(inputs: list):
    l = len(inputs)
    n = math.ceil(math.sqrt(l))
    fig = plt.figure("compiled show")
    for i in range(l):
        plt.subplot(n, n, i + 1), plt.imshow(inputs[i], 'gray'), plt.axis('off')
    return fig


if __name__ == '__main__':
    pass
