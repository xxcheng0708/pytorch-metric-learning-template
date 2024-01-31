from torch import nn
import torch
from torch.nn import functional as F


class STNFullyConv(nn.Module):
    def __init__(self):
        super(STNFullyConv, self).__init__()
        self.localization = nn.Sequential(nn.Conv2d(3, 16, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(16, 32, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(32, 64, kernel_size=3),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(64, 128, kernel_size=3),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True))

        self.conv_loc = nn.Sequential(nn.Conv2d(128, 6, kernel_size=3),
                                      nn.ReLU(True),
                                      nn.AdaptiveAvgPool2d((1, 1)))
        self.conv_loc[0].weight.data.zero_()
        self.conv_loc[0].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, x):
        # print("x.shape: {}".format(x.shape))
        xs = self.localization(x)
        # print(xs.shape)
        theta = self.conv_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


class STNFC(nn.Module):
    def __init__(self):
        super(STNFC, self).__init__()
        self.localization = nn.Sequential(nn.Conv2d(3, 8, kernel_size=7),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(8, 10, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True))
        self.fc_loc = nn.Sequential(nn.Linear(7840, 32),
                                    nn.ReLU(True),
                                    nn.Linear(32, 3 * 2))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0]))


    def forward(self, x):
        bs = x.shape[0]
        xs = self.localization(x)
        xs = xs.view(bs, -1)
        # print(xs.shape)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    x = torch.randn((8, 3, 128, 128)).cuda()

    model = STNFullyConv().cuda()
    res = model(x)
    print(res.shape)
    summary(model, input_size=(8, 3, 128, 128), batch_size=0)

    # model = STNFullyConv().cuda()
    # model = STNFC().cuda()
    # res = model(x)
    # print(res.shape)
    # summary(model, input_size=(8, 3, 128, 128), batch_size=0)
