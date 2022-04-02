# we cite the work of arXiv:1504.06375 [cs.CV]
import matplotlib.pyplot as plt
import numpy as np
import torch
import getopt
import numpy
import PIL
import PIL.Image
import cv2
import sys

from PicTrans2HEDInput import PicTrans2HEDInput


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.netVggOne = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggTwo = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggThr = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFou = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netVggFiv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=False)
        )

        self.netScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

        self.netCombine = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid()
        )

        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                              torch.hub.load_state_dict_from_url(
                                  url='http://content.sniklaus.com/github/pytorch-hed/network-' + arguments_strModel +
                                      '.pytorch',
                                  file_name='hed-' + arguments_strModel).items()})

    # end

    def forward(self, tenInput):
        tenInput = tenInput * 255.0
        tenInput = tenInput - torch.tensor(data=[104.00698793, 116.66876762, 122.67891434], dtype=tenInput.dtype,
                                           device=tenInput.device).view(1, 3, 1, 1)

        tenVggOne = self.netVggOne(tenInput)
        tenVggTwo = self.netVggTwo(tenVggOne)
        tenVggThr = self.netVggThr(tenVggTwo)
        tenVggFou = self.netVggFou(tenVggThr)
        tenVggFiv = self.netVggFiv(tenVggFou)

        tenScoreOne = self.netScoreOne(tenVggOne)
        tenScoreTwo = self.netScoreTwo(tenVggTwo)
        tenScoreThr = self.netScoreThr(tenVggThr)
        tenScoreFou = self.netScoreFou(tenVggFou)
        tenScoreFiv = self.netScoreFiv(tenVggFiv)

        tenScoreOne = torch.nn.functional.interpolate(input=tenScoreOne, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreTwo = torch.nn.functional.interpolate(input=tenScoreTwo, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreThr = torch.nn.functional.interpolate(input=tenScoreThr, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFou = torch.nn.functional.interpolate(input=tenScoreFou, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)
        tenScoreFiv = torch.nn.functional.interpolate(input=tenScoreFiv, size=(tenInput.shape[2], tenInput.shape[3]),
                                                      mode='bilinear', align_corners=False)

        return self.netCombine(torch.cat([tenScoreOne, tenScoreTwo, tenScoreThr, tenScoreFou, tenScoreFiv], 1))
    # end


# global network, only load the network once
HEDNetNetwork_GlobalVar = None


def estimate(tenInput):
    global HEDNetNetwork_GlobalVar

    # load network
    if HEDNetNetwork_GlobalVar is None:
        HEDNetNetwork_GlobalVar = Network().cuda().eval()
    # end

    intWidth = tenInput.shape[2]
    intHeight = tenInput.shape[1]

    print(tenInput.shape)

    assert (
            intWidth == 480)  # remember that there is no guarantee for correctness, comment this line out if you
    # acknowledge this and want to continue
    assert (
            intHeight == 320)  # remember that there is no guarantee for correctness, comment this line out if
    # you acknowledge this and want to continue

    return HEDNetNetwork_GlobalVar(tenInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()


# output function
# the pic must be 320(row)*480(col)*3(canal)
def HEDDetect(pic: np.ndarray):
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(
        pic[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0)))

    tenOutput = estimate(tenInput)

    output = (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)
    return output


def HEDDetectSave(pic: np.ndarray):
    tenInput = torch.FloatTensor(numpy.ascontiguousarray(
        pic[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                1.0 / 255.0)))

    tenOutput = estimate(tenInput)

    PIL.Image.fromarray(
        (tenOutput.clip(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(
        arguments_strOut)


# makes it available on terminal
if __name__ == '__main__':
    for i in range(200, 300):
        arguments_strModel = 'bsds500'  # only 'bsds500' for now
        # arguments_strIn = '../images/sample.png'
        arguments_strIn = '../../data/img_train_BrokenLine/23/draw.png'
        arguments_strOut = './out_1.png'

        ##########################################################

        assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)  # requires at least pytorch version 1.3.0

        torch.set_grad_enabled(False)  # make sure to not compute gradients for computational performance

        torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance
        pic = PicTrans2HEDInput(cv2.imread(arguments_strIn))
        HEDDetect(pic)
        input("stop:")
    pass
# end
