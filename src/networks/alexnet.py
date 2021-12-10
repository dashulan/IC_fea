import  torch
import  numpy as np
import  torch.nn as nn
def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False)
        # self.bn1 = nn.BatchNorm2d(64)
        s = compute_conv_output_size(32, 4)
        s = s // 2
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False)
        # self.bn2 = nn.BatchNorm2d(128)
        s = compute_conv_output_size(s, 3)
        s = s // 2
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(256, track_running_stats=False)
        # self.bn3 = nn.BatchNorm2d(256)
        s = compute_conv_output_size(s, 2)
        s = s // 2
        self.smid = s
        self.maxpool = torch.nn.MaxPool2d(2)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0.2)
        self.drop2 = torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * self.smid * self.smid, 2048, bias=False)
        self.bn4 = nn.BatchNorm1d(2048, track_running_stats=False)
        # self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 2048, bias=False)
        self.bn5 = nn.BatchNorm1d(2048, track_running_stats=False)
        # self.bn5 = nn.BatchNorm1d(2048)

        self.bnL = nn.BatchNorm1d(2048)

        self.fc = torch.nn.Linear(2048,10)
        # self.taskcla = taskcla
        # self.fc3 = torch.nn.ModuleList()
        # for t, n in self.taskcla:
        #     self.fc3.append(torch.nn.Linear(2048, n, bias=False))

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, returnFea=False):
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1(x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2(x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3(x))))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4(x)))

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5(x)))
        x = self.bnL(x)
        x = self.fc(x)
        return x
        # y = []
        #
        # for t, i in self.taskcla:
        #     y.append(self.fc3[t](x))
        # if returnFea:
        #     return y, x
        # return y