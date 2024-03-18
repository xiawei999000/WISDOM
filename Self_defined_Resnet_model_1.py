import os
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class BasicBlock(nn.Module):
    def __init__(self, in_channel, s):
        super(BasicBlock, self).__init__()
        self.s = s
        self.conv1 = nn.Conv2d(in_channel, in_channel * s, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel * s)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel * s, in_channel * s, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channel * s)
        if self.s == 2:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, in_channel * s, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(in_channel * s)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.s == 2:  # 缩小
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class ResNetSelf_Img(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNetSelf_Img, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        # remove the layers for small image with 32*32
        # should be modified forward
        #
        # self.layer3 = nn.Sequential(
        #     BasicBlock(in_channel=128, s=2),
        #     BasicBlock(in_channel=256, s=1),
        # )
        # self.layer4 = nn.Sequential(
        #     BasicBlock(in_channel=256, s=2),
        #     BasicBlock(in_channel=512, s=1),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, n_class)
        self.sigmoid_img = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid_img(x)
        return x


# multiInputOutput
class ResNetSelf_Combine_TwoOuts(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNetSelf_Combine_TwoOuts, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        # remove the layers for small image with 32*32
        # should be modified forward
        #
        # self.layer3 = nn.Sequential(
        #     BasicBlock(in_channel=128, s=2),
        #     BasicBlock(in_channel=256, s=1),
        # )
        # self.layer4 = nn.Sequential(
        #     BasicBlock(in_channel=256, s=2),
        #     BasicBlock(in_channel=512, s=1),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, n_class)
        self.fc2 = nn.Linear(5, 5)  # in :patch_pred + LD SD RD ADC
        self.fc3 = nn.Linear(5, n_class)  # out: prob
        self.sigmoid_img = nn.Sigmoid()
        self.sigmoid_combined = nn.Sigmoid()


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, patch, patch_LD, patch_SD, patch_RD, patch_adc):
        x = self.conv1(patch)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        img_pred = self.fc1(x)
        img_pred = self.sigmoid_img(img_pred)

        combined_fea = torch.cat((img_pred[0], patch_LD, patch_SD, patch_RD, patch_adc))
        x2 = self.fc2(combined_fea)
        combined_pred = self.fc3(x2)
        combined_pred = self.sigmoid_combined(combined_pred)
        combined_pred = combined_pred.unsqueeze(0)

        return img_pred, combined_pred


class ResNetSelf_Combine_ImgFeas_TwoOuts(nn.Module):
    def __init__(self, n_class, zero_init_residual=True):
        super(ResNetSelf_Combine_ImgFeas_TwoOuts, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            BasicBlock(in_channel=64, s=1),
            BasicBlock(in_channel=64, s=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(in_channel=64, s=2),
            BasicBlock(in_channel=128, s=1),
        )
        # remove the layers for small image with 32*32
        # should be modified forward
        self.layer3 = nn.Sequential(
            BasicBlock(in_channel=128, s=2),
            BasicBlock(in_channel=256, s=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channel=256, s=1),
            BasicBlock(in_channel=256, s=1),
        )
        ###################################################
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_imgFeas = nn.Linear(256, 4) # ImgFeas
        self.fc_imgProb = nn.Linear(4, 1)
        self.sigmoid_img = nn.Sigmoid()

        self.fc_Combine = nn.Linear(8, 4)  # in : 4ImgFeas + LD SD RD ADC = 8 feas  out : combined to 4 feas
        self.fc_combine_2 = nn.Linear(4, n_class)  # out: 4 -> 1 prob
        self.sigmoid_combined = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, patch, patch_LD, patch_SD, patch_RD, patch_adc):
        x = self.conv1(patch)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        img_feas = self.fc_imgFeas(x)
        img_feas = self.relu(img_feas)  # RELU
        img_pred = self.fc_imgProb(img_feas)
        img_pred = self.sigmoid_img(img_pred)

        combined_fea = torch.cat((img_feas[0], patch_LD, patch_SD, patch_RD, patch_adc))
        combined_fea = self.fc_Combine(combined_fea)
        combined_fea = self.relu(combined_fea)  # RELU
        combined_pred = self.fc_combine_2(combined_fea)
        combined_pred = self.sigmoid_combined(combined_pred)
        combined_pred = combined_pred.unsqueeze(0)

        return img_pred, combined_pred
