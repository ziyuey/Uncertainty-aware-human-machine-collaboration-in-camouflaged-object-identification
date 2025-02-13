import os
import h5py
import math
import numpy as np
import torch, gc
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn, Tensor

class SepConv(torch.nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()

        self.relu = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding,
                                     groups=C_in,
                                     bias=False)
        self.pw = torch.nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False)
        self.bn = torch.nn.BatchNorm2d(C_out, eps=1e-3, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv1(x)
        x = self.pw(x)
        x = self.bn(x)
        return x


class channel_se(torch.nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(channel_se, self).__init__()

        # Attribute assignment
        # avgpooling,The output weight and high =1
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)

        # the first FC reduce the channel to the 1/4
        self.fc1 = torch.nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)

        # RELU to activate
        self.relu = torch.nn.ReLU()

        # the second fc to recover the channel
        self.fc2 = torch.nn.Linear(in_features=in_channel // (ratio), out_features=in_channel, bias=False)

        # sigmoid activate limit the weight between 0 and 1
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):  # the input is the feature
        b, n, c, t = inputs.shape

        # [b,n,c,t]==>[b,c,n,t]
        inputs = inputs.reshape(b, c, n, t)

        # pooling [b,c,n,t] ==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # first [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)

        # second fc [b,c//4]==>[b,c]
        x = self.fc2(x)
        x = self.sigmoid(x)

        # [b,c] ==> [b,c,1,1]
        x = x.view([b, c, 1, 1])

        outputs = x * inputs
        outputs = outputs.reshape(b, n, c, t)

        return outputs


class PLNet(torch.nn.Module):  # EEGNet 8-2
    def __init__(self, F1=8, F2=16, D=2, Channel=4, T=251, Dropoutrate=0.5, N=2):
        super(PLNet, self).__init__()

        self.conv2D1 = torch.nn.Conv2d(1, F1, (1, 64), stride=1, padding='same', bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.depthwiseconv2D1 = torch.nn.Conv2d(F1, D * F1, (Channel, 1))
        # self.depthwiseconv2D1 = SepConv(F1, F1 * D, (Channel, 1), stride=1, padding=0)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=D * F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.activate1 = torch.nn.ELU()
        self.avgpool2D1 = torch.nn.AvgPool2d((1, 4))
        self.dropout1 = torch.nn.Dropout(Dropoutrate)

        self.conv2D2 = SepConv(D * F1, F2, (1, 16), stride=1, padding='same')
        # self.batchnorm3 = torch.nn.BatchNorm2d(num_features=F2, eps=1e-3,
        #                                        momentum=0.01, affine=True)
        self.activate2 = torch.nn.ELU()
        self.avgpool2D2 = torch.nn.AvgPool2d((1, 4))
        self.dropout2 = torch.nn.Dropout(Dropoutrate)

        self.liner = torch.nn.Linear(320, N)

        # PLNet
        self.PL_Conv2D = nn.Conv2d(1, 8, (1, 64), (1, 4))
        self.PL_batchnorm1 = torch.nn.BatchNorm2d(num_features=8, eps=1e-3,
                                                  momentum=0.01, affine=True)
        self.PL_activation = torch.nn.Linear(47, 47)
        self.PL_depthwiseConv2D = nn.Conv2d(47, 47, (Channel, 1), (1, 1), groups=47)
        self.PL_batchnorm2 = torch.nn.BatchNorm2d(num_features=8, eps=1e-3,
                                                  momentum=0.01, affine=True)
        self.PL_SeparableConv2D = torch.nn.Sequential(
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(8, 8, kernel_size=(1, 32), stride=(1, 1),
                            groups=8,
                            bias=False),
            torch.nn.Conv2d(8, 16, kernel_size=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(16, eps=1e-3, affine=True))
        self.PL_averaging = torch.nn.AvgPool2d((1, 4))
        self.PL_liner = torch.nn.Linear(3776, N)

    def forward(self, x):
        # PLNet
        x = x[:, :, :62, :]
        x = self.PL_Conv2D(x)
        x = self.PL_batchnorm1(x)
        x = self.PL_activation(x)
        x = x.permute(0, 3, 2, 1)
        x = self.PL_depthwiseConv2D(x)
        x = x.permute(0, 3, 2, 1)
        x = self.PL_batchnorm2(x)
        x = self.activate1(x)
        x = self.dropout1(x)

        x = self.PL_SeparableConv2D(x)
        # x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.PL_averaging(x)
        x = self.dropout2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.PL_liner(x)

        return x

class EEGNet(torch.nn.Module):  # EEGNet 8-2
    def __init__(self, F1=8, F2=16, D=2, Channel=4, T=251, Dropoutrate=0.5, N=2):
        super(EEGNet, self).__init__()

        self.conv2D1 = torch.nn.Conv2d(1, F1, (1, 64), stride=1, padding='same', bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.depthwiseconv2D1 = torch.nn.Conv2d(F1, D * F1, (Channel, 1))
        # self.depthwiseconv2D1 = SepConv(F1, F1 * D, (Channel, 1), stride=1, padding=0)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=D * F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.activate1 = torch.nn.ELU()
        self.avgpool2D1 = torch.nn.AvgPool2d((1, 2))
        self.dropout1 = torch.nn.Dropout(Dropoutrate)

        self.conv2D2 = SepConv(D * F1, F2, (1, 16), stride=1, padding='same')
        # self.batchnorm3 = torch.nn.BatchNorm2d(num_features=F2, eps=1e-3,
        #                                        momentum=0.01, affine=True)
        self.activate2 = torch.nn.ELU()
        self.avgpool2D2 = torch.nn.AvgPool2d((1, 4))
        self.dropout2 = torch.nn.Dropout(Dropoutrate)

        self.liner = torch.nn.Linear(29264, N)

    def forward(self, x):
        # EEGNet
        x = self.conv2D1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseconv2D1(x)
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.avgpool2D1(x)
        x = self.dropout1(x)

        x = self.conv2D2(x)
        x = self.activate2(x)
        x = self.avgpool2D2(x)
        x = self.dropout2(x)

        x = x.reshape(x.shape[0], -1)
        xx = x
        x = self.liner(x)

        return x

class PPNN(torch.nn.Module):  # EEGNet 8-2
    def __init__(self, F1=8, F2=16, D=2, Channel=4, T=251, Dropoutrate=0.5, N=2):
        super(PPNN, self).__init__()

        self.conv2D1 = torch.nn.Conv2d(1, F1, (1, 64), stride=1, padding='same', bias=False)
        self.batchnorm1 = torch.nn.BatchNorm2d(num_features=F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.depthwiseconv2D1 = torch.nn.Conv2d(F1, D * F1, (Channel, 1))
        # self.depthwiseconv2D1 = SepConv(F1, F1 * D, (Channel, 1), stride=1, padding=0)
        self.batchnorm2 = torch.nn.BatchNorm2d(num_features=D * F1, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.activate1 = torch.nn.ELU()
        self.avgpool2D1 = torch.nn.AvgPool2d((1, 4))
        self.dropout1 = torch.nn.Dropout(Dropoutrate)

        self.conv2D2 = SepConv(D * F1, F2, (1, 16), stride=1, padding='same')
        # self.batchnorm3 = torch.nn.BatchNorm2d(num_features=F2, eps=1e-3,
        #                                        momentum=0.01, affine=True)
        self.activate2 = torch.nn.ELU()
        self.avgpool2D2 = torch.nn.AvgPool2d((1, 4))
        self.dropout2 = torch.nn.Dropout(Dropoutrate)

        self.liner = torch.nn.Linear(320, N)

        # PPNN
        self.PPNN_conv2D1 = torch.nn.Conv2d(1, F1, (1, 3), stride=1, padding='same', bias=False, dilation=(1, 2))
        self.PPNN_conv2D2 = torch.nn.Conv2d(F1, F1, (1, 3), stride=1, padding='same', bias=False, dilation=(1, 4))
        self.PPNN_conv2D3 = torch.nn.Conv2d(F1, F1, (1, 3), stride=1, padding='same', bias=False, dilation=(1, 8))
        self.PPNN_conv2D4 = torch.nn.Conv2d(F1, F1, (1, 3), stride=1, padding='same', bias=False, dilation=(1, 16))
        self.PPNN_conv2D5 = torch.nn.Conv2d(F1, F1, (1, 3), stride=1, padding='same', bias=False, dilation=(1, 32))
        self.PPNN_conv2D6 = torch.nn.Conv2d(F1, F2, (Channel, 1))
        self.PP_liner = torch.nn.Linear(236944, N)

    def forward(self, x):
        # PPNN
        x = self.PPNN_conv2D1(x)
        x = self.PPNN_conv2D2(x)
        x = self.PPNN_conv2D3(x)
        x = self.PPNN_conv2D4(x)
        x = self.PPNN_conv2D5(x)
        x = self.batchnorm1(x)
        x = self.activate1(x)
        x = self.PPNN_conv2D6(x)
        x = self.batchnorm2(x)
        x = self.activate2(x)
        x = self.dropout2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.PP_liner(x)

        return x


def weights_gen_func(channel_input, filter_size):
    pad_size = (filter_size - 1) // 2
    weights_gen_ins = nn.Sequential(
        nn.ZeroPad2d((0, 0, pad_size, pad_size)),
        nn.Conv2d(channel_input * 2, channel_input * 1, (filter_size, 1)),
        nn.ReLU(inplace=True),
        nn.ZeroPad2d((0, 0, pad_size, pad_size)),
        nn.Conv2d(channel_input * 1, channel_input * 1, (filter_size, 1)),
        nn.Sigmoid(),
    )
    return weights_gen_ins


class MMR(nn.Module):
    def __init__(self, channel_input=128, head=4):
        super(MMR, self).__init__()
        self.head = head

        self.ada_func_eeg = nn.ModuleList([weights_gen_func(channel_input // head, filter_size=2 * i + 1)
                                           for i in range(head)])
        self.ada_func_eye = nn.ModuleList([weights_gen_func(channel_input // head, filter_size=2 * i + 1)
                                           for i in range(head)])
        self.output = nn.Sequential(
            nn.Linear(20736, 5),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_eeg, x_eye):
        x_fusion_in_eeg = x_eeg  # [256, 128, 94, 1]
        x_fusion_in_eye = x_eye  # [256, 128, 94, 1]
        all_channel = x_fusion_in_eeg.size(1)  # 128
        head_channel = all_channel // self.head  # 128 / 4 = 32
        x_eeg_part = [x_fusion_in_eeg[:, i * head_channel: (i + 1) * head_channel, :, :] for i in
                      range(self.head)]  # 4 * [256, 32, 94, 1]
        x_eye_part = [x_fusion_in_eye[:, i * head_channel: (i + 1) * head_channel, :, :] for i in
                      range(self.head)]  # 4 * [256, 32, 94, 1]
        x_fusion_in_part = [torch.cat([x_eeg_part[i], x_eye_part[i]], dim=1)
                            for i in range(self.head)]  # 4 * [256, 64, 94, 1]
        x_eeg_maps = [self.ada_func_eeg[i](x_fusion_in_part[i]) for i in range(self.head)]  # 4 * [256, 32, 94, 1]
        x_eye_maps = [self.ada_func_eye[i](x_fusion_in_part[i]) for i in range(self.head)]  # 4 * [256, 32, 94, 1]
        x_eeg_rec = [x_eeg_part[i] * (x_eeg_maps[i]) for i in range(self.head)]  # [256, 32, 94, 1]
        x_eye_rec = [x_eye_part[i] * (x_eye_maps[i]) for i in range(self.head)]  # [256, 32, 94, 1]
        x_concat_eeg = torch.cat(x_eeg_rec, dim=1)  # [256, 128, 94, 1]
        x_concat_eye = torch.cat(x_eye_rec, dim=1)  # [256, 128, 94, 1]
        x_concat = torch.cat([x_concat_eeg, x_concat_eye], dim=1)  # [256, 256, 94, 1]
        x_concat = x_concat.reshape(x_concat.size(0), -1)  # [256, 24064]
        x_concat = torch.tanh(x_concat) * torch.sigmoid(x_concat)  # [256, 24064]

        output = self.output(x_concat)

        # addition_output = (x_eeg_maps, x_eye_maps)
        return output


class FeatureGuiding(nn.Module):
    def __init__(self, c):
        super(FeatureGuiding, self).__init__()
        filter_size = 3
        padding_size = (filter_size - 1) // 2
        self.modulate_gamma = nn.Sequential(
            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.Sigmoid()
        )
        self.modulate_beta = nn.Sequential(
            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ZeroPad2d((0, 0, padding_size, padding_size)),
            nn.Conv2d(c, c, (filter_size, 1)),
            nn.Sigmoid()
        )

    def forward(self, x_out_eeg, x_out_eye):
        x_in_eeg = x_out_eeg
        x_in_eye = x_out_eye
        weight_gamma = self.modulate_gamma(x_in_eeg)
        weight_beta = self.modulate_beta(x_in_eeg)
        x_output_eye = x_in_eye * weight_gamma + weight_beta
        return x_output_eye


class CMFG(nn.Module):
    def __init__(self, eeg_feature_size=62, eye_feature_size=4):
        super(CMFG, self).__init__()
        self.conv1_eeg = nn.Sequential(
            nn.Conv2d(1, 64, (1, eeg_feature_size), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_eeg = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(64, 96, (3, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.conv3_eeg = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(96, 128, (3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        self.conv1_eye = nn.Sequential(
            nn.Conv2d(1, 64, (1, eye_feature_size), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2_eye = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(64, 96, (3, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.conv3_eye = nn.Sequential(
            nn.ZeroPad2d((0, 0, 1, 1)),
            nn.Conv2d(96, 128, (3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.transfer_1 = FeatureGuiding(64)
        self.transfer_2 = FeatureGuiding(96)
        self.transfer_3 = FeatureGuiding(128)
        self.fusion_func = MMR()
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.mha2 = torch.nn.MultiheadAttention(embed_dim=128, num_heads=4)

    def forward(self, x):
        # x_eeg:[batch_size, 64, 64], x_eye:[batch_size, 64, 6]            [BatchSize, T, C]

        x_eeg = x[:, :, :62, :]  # [256, 62, 376] [b, c, t]
        x_eye = x[:, :, 62:, :]  # [256, 4, 376] [b, c, t]
        x_eeg = x_eeg.transpose(2, 3)
        x_eye = x_eye.transpose(2, 3)

        # x_eeg = torch.unsqueeze(x_eeg, dim=1)  # [256, 1, 376, 62]
        # x_eye = torch.unsqueeze(x_eye, dim=1)  # [256, 1, 376, 4]
        # (BatchSize, Channel, T, C)

        x_1_eeg = self.conv1_eeg(x_eeg)  # [256, 64, 376, 1]
        x_1_eye = self.conv1_eye(x_eye)  # [256, 64, 376, 1]

        # x_2_eeg_in = x_1_eeg  # [256, 64, 376, 1]
        # x_2_eye_in = self.transfer_1(x_1_eeg, x_1_eye)  # [256, 64, 376, 1]
        x_2_eeg_in = self.transfer_1(x_1_eye, x_1_eeg)  # [256, 64, 376, 1]
        x_2_eye_in = x_1_eye  # [256, 64, 376, 1]

        x_2_eeg = self.conv2_eeg(x_2_eeg_in)  # [256, 96, 188, 1]
        x_2_eye = self.conv2_eye(x_2_eye_in)  # [256, 96, 188, 1]

        # x_3_eeg_in = x_2_eeg  # [256, 96, 188, 1]
        # x_3_eye_in = self.transfer_2(x_2_eeg, x_2_eye)  # [256, 96, 188, 1]
        x_3_eeg_in = self.transfer_2(x_2_eye, x_2_eeg)  # [256, 96, 188, 1]
        x_3_eye_in = x_2_eye  # [256, 96, 188, 1]

        x_3_eeg = self.conv3_eeg(x_3_eeg_in)  # [256, 128, 94, 1]
        x_3_eye = self.conv3_eye(x_3_eye_in)  # [256, 128, 94, 1]

        # x_4_eeg_in = x_3_eeg  # [256, 128, 94, 1]
        # x_4_eye_in = self.transfer_3(x_3_eeg, x_3_eye)  # [256, 128, 94, 1]
        x_4_eeg_in = self.transfer_3(x_3_eye, x_3_eeg)  # [256, 128, 94, 1]
        x_4_eye_in = x_3_eye  # [256, 128, 94, 1]
        # x_4_eeg_in = x_4_eeg_in.squeeze(3)
        # x_4_eeg_in = x_4_eeg_in.transpose(1, 2)
        # x_4_eye_in = x_4_eye_in.squeeze(3)
        # x_4_eye_in = x_4_eye_in.transpose(1, 2)
        # x3, _ = self.mha1(x_4_eeg_in, x_4_eye_in, x_4_eye_in)
        # x4, _ = self.mha2(x_4_eye_in, x_4_eeg_in, x_4_eeg_in)
        # x3 = x3.transpose(1, 2).unsqueeze(3)
        # x4 = x4.transpose(1, 2).unsqueeze(3)
        # x_4_eeg_in = x_4_eeg_in.transpose(1, 2).unsqueeze(3)
        # x_4_eye_in = x_4_eye_in.transpose(1, 2).unsqueeze(3)

        x_out_eeg = x_4_eeg_in  # [128, 128, 81, 1]
        x_out_eye = x_4_eye_in  # [128, 128, 81, 1]

        output = self.fusion_func(x_out_eeg, x_out_eye)

        return output

class TransEEGNet(torch.nn.Module):
    def __init__(self, channels=62):
        super(TransEEGNet, self).__init__()
        self.mha1 = torch.nn.MultiheadAttention(embed_dim=62, num_heads=2)
        self.mha2 = torch.nn.MultiheadAttention(embed_dim=4, num_heads=2)
        self.mha3 = torch.nn.MultiheadAttention(embed_dim=66, num_heads=2)
        self.mha4 = torch.nn.MultiheadAttention(embed_dim=32, num_heads=2)
        self.mha5 = torch.nn.MultiheadAttention(embed_dim=16, num_heads=2)
        self.mha6 = torch.nn.MultiheadAttention(embed_dim=16, num_heads=2)
        self.eegnet1 = EEGnet_PLNet_PPNN(Channel=62)
        self.eegnet2 = EEGnet_PLNet_PPNN(Channel=62)
        self.eegnet3 = EEGnet_PLNet_PPNN(Channel=66)
        self.conv1 = torch.nn.Conv2d(32, 16, (1, 3), padding='same')
        self.elu1 = torch.nn.ELU()
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, (1, 3), padding='same')
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(16, 32, (1, 1), padding='same')
        self.sig1 = torch.nn.Sigmoid()
        self.liner1 = torch.nn.Linear(128, 64)
        self.elu2 = torch.nn.ELU()
        self.liner2 = torch.nn.Linear(64, 5)

        self.se1 = channel_se(62)
        self.se1 = channel_se(4)
        self.head = 2

        self.ada_func_eeg = nn.ModuleList([weights_gen_func(16 // 2, filter_size=2 * i + 1)
                                           for i in range(2)])
        self.ada_func_eye = nn.ModuleList([weights_gen_func(16 // 2, filter_size=2 * i + 1)
                                           for i in range(2)])
        self.output = nn.Sequential(
            nn.Linear(16 * 40, 5),
        )
        self.softmax = nn.Softmax(dim=1)

        self.flf = torch.nn.Linear(272, 2)
        self.flf2 = torch.nn.Linear(272, 2)

    def forward(self, x, mod):
        # x = x.squeeze(1)
        # x = x.transpose(1, 2)
        # x, _ = self.mha3(x, x, x)
        # x = x.transpose(1, 2).unsqueeze(1)
        # x = self.eegnet3(x)
        if mod == 1:
            x = x[:, :, :62, :]
            x = self.eegnet1(x)
            x = x.reshape(x.shape[0], -1)
            x = self.flf(x)

        if mod == 2:
            x = x[:, :, 62:, :]
            x = self.eegnet2(x)
            x = x.reshape(x.shape[0], -1)
            x = self.flf2(x)

        if mod == 3:
            x1 = x[:, :, :62, :]
            x1 = self.eegnet1(x1)
            x2 = x[:, :, 62:, :]
            x2 = self.eegnet2(x2)
            x3 = x1 + x2
            x3 = x3.reshape(x3.shape[0], -1)
            x = self.flf(x3)

        #
        # x1 = self.eegnet1(x1)
        # # x1 = x1.transpose(2, 3)
        # #
        # # # x2 = x2.squeeze(1)
        # # # x2 = x2.transpose(1, 2)
        # # # x2, _ = self.mha2(x2, x2, x2)
        # # # x2 = x2.transpose(1, 2).unsqueeze(1)
        # # # x2 = x2 + x[:, :, 62:, :]
        # x2 = self.eegnet2(x2)
        # x2 = x2.squeeze(2)
        # x2 = x2.transpose(1, 2)
        # x1 = x1.squeeze(2)
        # x1 = x1.transpose(1, 2)
        # x2, _ = self.mha5(x2, x1, x1)
        # x2 = x2.transpose(1, 2).unsqueeze(2)
        # x1 = x1.transpose(1, 2).unsqueeze(2)
        # # x2 = x2.transpose(2, 3)
        #
        # x3 = torch.cat((x1, x2), dim=1)
        # # # # x3 = x3.squeeze(2)
        # # # # x3 = x3.transpose(1, 2)
        # # # # x3, _ = self.mha4(x3, x3, x3)
        # # # # x3 = x3.transpose(1, 2).unsqueeze(2)
        # # #
        # x3 = self.conv1(x3)
        # x3 = self.relu1(x3)
        # x3 = self.conv2(x3)
        # # # x3 = self.relu2(x3)
        # # # x3 = self.conv3(x3)
        # x3 = self.sig1(x3)
        # x1_weight = x3[:, :16, :, :]
        # x2_weight = x3[:, 16:, :, :]
        # x1 = torch.mul(x1, x1_weight)
        # x2 = torch.mul(x2, x2_weight)
        # # # x = x1_weight + x2_weight + x1 + x2
        # # #
        # # # # 版本1
        # # x1 = x[:, :, :62, :]
        # # x2 = x[:, :, 62:, :]
        # #
        # # x1 = self.eegnet1(x1)
        # # x2 = self.eegnet2(x2)
        # # x3 = x1 + x2
        # # x3 = x3.reshape(x3.shape[0], -1)
        # # x = self.flf(x3)
        # #
        # x1 = x1.squeeze(2)
        # x1 = x1.transpose(1, 2)
        # x2 = x2.squeeze(2)
        # x2 = x2.transpose(1, 2)
        # x3, _ = self.mha5(x1, x2, x2)
        # x4, _ = self.mha6(x2, x1, x1)
        # x1 = x1.transpose(1, 2).unsqueeze(2)
        # x2 = x2.transpose(1, 2).unsqueeze(2)
        # x3 = x3.transpose(1, 2).unsqueeze(2)
        # x4 = x4.transpose(1, 2).unsqueeze(2)

        # x = x1
        #
        # x = x.reshape(x.shape[0], -1)
        # # print(x.shape)
        # x = self.liner1(x)
        # x = self.elu2(x)
        # x = self.liner2(x)
        # x = self.flf(x)

        return x

class EEGInception(torch.nn.Module):
    def __init__(self):
        super(EEGInception, self).__init__()
        self.Conv2D1_1 = torch.nn.Conv2d(1, 8, (1, 64), stride=1, padding='same', bias=False)
        self.BatchNorm1_1 = torch.nn.BatchNorm2d(num_features=8, eps=1e-3,
                                                 momentum=0.01, affine=True)
        self.activate = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(0.25)
        self.DepthWiseConv2D1_1 = torch.nn.Conv2d(8, 16, (62, 1), stride=1, bias=False)
        self.BatchNorm1_2 = torch.nn.BatchNorm2d(num_features=16, eps=1e-3,
                                                 momentum=0.01, affine=True)
        self.Conv2D1_2 = torch.nn.Conv2d(1, 8, (1, 32), stride=1, padding='same', bias=False)
        self.DepthWiseConv2D1_2 = torch.nn.Conv2d(8, 16, (62, 1), stride=1, bias=False)
        self.Conv2D1_3 = torch.nn.Conv2d(1, 8, (1, 16), stride=1, padding='same', bias=False)
        self.DepthWiseConv2D1_3 = torch.nn.Conv2d(8, 16, (62, 1), stride=1, bias=False)
        self.AvgPool2D1 = torch.nn.AvgPool2d((1, 4))

        self.Conv2D2_1 = torch.nn.Conv2d(48, 8, (1, 16), stride=1, padding='same', bias=False)
        self.BatchNorm2 = torch.nn.BatchNorm2d(num_features=8, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.Conv2D2_2 = torch.nn.Conv2d(48, 8, (1, 8), stride=1, padding='same', bias=False)
        self.Conv2D2_3 = torch.nn.Conv2d(48, 8, (1, 4), stride=1, padding='same', bias=False)
        self.AvgPool2D2 = torch.nn.AvgPool2d((1, 2))

        self.Conv2D3 = torch.nn.Conv2d(24, 12, (1, 8), stride=1, padding='same', bias=False)
        self.BatchNorm3 = torch.nn.BatchNorm2d(num_features=12, eps=1e-3,
                                               momentum=0.01, affine=True)
        self.Conv2D4 = torch.nn.Conv2d(12, 6, (1, 4), stride=1, padding='same', bias=False)
        self.BatchNorm4 = torch.nn.BatchNorm2d(num_features=6, eps=1e-3,
                                               momentum=0.01, affine=True)

        self.liner = torch.nn.Linear(42, 5)

    def forward(self, x):
        x = x[:, :, :62, :]
        x1 = self.Conv2D1_1(x)
        x1 = self.BatchNorm1_1(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)
        x1 = self.DepthWiseConv2D1_1(x1)
        x1 = self.BatchNorm1_2(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x2 = self.Conv2D1_2(x)
        x2 = self.BatchNorm1_1(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)
        x2 = self.DepthWiseConv2D1_2(x2)
        x2 = self.BatchNorm1_2(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)

        x3 = self.Conv2D1_3(x)
        x3 = self.BatchNorm1_1(x3)
        x3 = self.activate(x3)
        x3 = self.dropout(x3)
        x3 = self.DepthWiseConv2D1_3(x3)
        x3 = self.BatchNorm1_2(x3)
        x3 = self.activate(x3)
        x3 = self.dropout(x3)

        x = torch.cat((x1, x2, x3), 1)
        x = self.AvgPool2D1(x)

        x1 = self.Conv2D2_1(x)
        x1 = self.BatchNorm2(x1)
        x1 = self.activate(x1)
        x1 = self.dropout(x1)

        x2 = self.Conv2D2_2(x)
        x2 = self.BatchNorm2(x2)
        x2 = self.activate(x2)
        x2 = self.dropout(x2)

        x3 = self.Conv2D2_3(x)
        x3 = self.BatchNorm2(x3)
        x3 = self.activate(x3)
        x3 = self.dropout(x3)

        x = torch.cat((x1, x2, x3), 1)
        x = self.AvgPool2D2(x)

        x = self.Conv2D3(x)
        x = self.BatchNorm3(x)
        x = self.activate(x)
        x = self.AvgPool2D2(x)
        x = self.dropout(x)
        # x = self.AvgPool2D2(x)

        x = self.Conv2D4(x)
        x = self.BatchNorm4(x)
        x = self.activate(x)
        x = self.AvgPool2D2(x)
        x = self.dropout(x)
        # x = self.AvgPool2D2(x)

        x = x.reshape(x.shape[0], -1)
        x = self.liner(x)

        return x

class EEGDepthAttention(nn.Module):

    def __init__(self, W, C, k=7):
        super(EEGDepthAttention, self).__init__()
        self.C = C
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, W))
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(k // 2, 0), bias=True)  # original kernel k
        self.softmax = nn.Softmax(dim=-2)

    def forward(self, x):
        x_pool = self.adaptive_pool(x)
        x_transpose = x_pool.permute(0, 2, 3, 1)
        y = self.conv(x_transpose)
        y = self.softmax(y)
        y = y.permute(0, 3, 1, 2)

        if y.shape[2:] != x.shape[2:]:
            y = F.interpolate(y, size=x.shape[2:], mode='nearest')

        result = y * self.C * x
        return result


class LMDA(nn.Module):
    """
    LMDA-Net for the paper
    """

    def __init__(self, chans=62, samples=326, num_classes=2, depth=9, kernel=75, channel_depth1=24, channel_depth2=9,
                 ave_depth=1, avepool=5):
        super(LMDA, self).__init__()
        self.ave_depth = ave_depth
        self.channel_weight = nn.Parameter(torch.randn(depth, 1, chans), requires_grad=True)
        nn.init.xavier_uniform_(self.channel_weight.data)
        # nn.init.kaiming_normal_(self.channel_weight.data, nonlinearity='relu')
        # nn.init.normal_(self.channel_weight.data)
        # nn.init.constant_(self.channel_weight.data, val=1/chans)

        self.time_conv = nn.Sequential(
            nn.Conv2d(depth, channel_depth1, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.Conv2d(channel_depth1, channel_depth1, kernel_size=(1, kernel),
                      groups=channel_depth1, bias=False),
            nn.BatchNorm2d(channel_depth1),
            nn.GELU(),
        )
        # self.avgPool1 = nn.AvgPool2d((1, 24))
        self.chanel_conv = nn.Sequential(
            nn.Conv2d(channel_depth1, channel_depth2, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.Conv2d(channel_depth2, channel_depth2, kernel_size=(chans, 1), groups=channel_depth2, bias=False),
            nn.BatchNorm2d(channel_depth2),
            nn.GELU(),
        )

        self.norm = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            # nn.AdaptiveAvgPool3d((9, 1, 35)),
            nn.Dropout(p=0.65),
        )

        # 定义自动填充模块
        out = torch.ones((1, 1, chans, samples))
        out = torch.einsum('bdcw, hdc->bhcw', out, self.channel_weight)
        out = self.time_conv(out)
        # out = self.avgPool1(out)
        N, C, H, W = out.size()

        self.depthAttention = EEGDepthAttention(W, C, k=7)

        out = self.chanel_conv(out)
        out = self.norm(out)
        n_out_time = out.cpu().data.numpy().shape
        print('In ShallowNet, n_out_time shape: ', n_out_time)
        self.classifier = nn.Linear(315, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x[:, :, :62, :]
        x = torch.einsum('bdcw, hdc->bhcw', x, self.channel_weight)  # 导联权重筛选

        x_time = self.time_conv(x)  # batch, depth1, channel, samples_
        x_time = self.depthAttention(x_time)  # DA1

        x = self.chanel_conv(x_time)  # batch, depth2, 1, samples_
        x = self.norm(x)

        features = torch.flatten(x, 1)
        cls = self.classifier(features)
        return cls


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (62, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        # print(x.shape)
        out = self.fc(x)
        return out

class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=2, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )

class CMGF(torch.nn.Module):
    def __init__(self, channels=62):
        super(CMGF, self).__init__()

        self.eeg_block1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 62), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.eeg_block2 = nn.Sequential(
            nn.Conv2d(64, 96, (3, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.eeg_block3 = nn.Sequential(
            nn.Conv2d(96, 128, (3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        self.eye_block1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 4), padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.eye_block2 = nn.Sequential(
            nn.Conv2d(64, 96, (3, 1)),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )
        self.eye_block3 = nn.Sequential(
            nn.Conv2d(96, 128, (3, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1))
        )

        self.eeg_guid1 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.eeg_guid2 = nn.Sequential(
            nn.Conv2d(96, 96, (3, 1), padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.eeg_guid3 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), padding='same'),
            nn.Sigmoid()
        )
        self.eeg_guid4 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 1), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.eeg_guid5 = nn.Sequential(
            nn.Conv2d(96, 96, (3, 1), padding='same'),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.eeg_guid6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), padding='same'),
            nn.Sigmoid()
        )
        self.conv1 = torch.nn.Conv2d(256, 256, (5, 1), padding='same')
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(256, 256, (5, 1), padding='same')
        self.sig1 = torch.nn.Sigmoid()
        self.liner1 = torch.nn.Linear(10240, 5)

    def forward(self, x):
        eeg = x[:, :, :62, :]
        eeg = eeg.transpose(2, 3)
        eye = x[:, :, 62:, :]
        eye = eye.transpose(2, 3)
        eeg = self.eeg_block1(eeg)
        eye = self.eye_block1(eye)

        guid1 = self.eeg_guid1(eeg)
        guid4 = self.eeg_guid4(eeg)
        eye = torch.mul(eye, guid1) + guid4

        eeg = self.eeg_block2(eeg)
        eye = self.eye_block2(eye)

        guid2 = self.eeg_guid2(eeg)
        guid5 = self.eeg_guid5(eeg)
        eye = torch.mul(eye, guid2) + guid5

        eeg = self.eeg_block3(eeg)
        eye = self.eye_block3(eye)

        guid3 = self.eeg_guid3(eeg)
        guid6 = self.eeg_guid6(eeg)
        eye = torch.mul(eye, guid3) + guid6

        x3 = torch.cat((eeg, eye), 1)
        x3 = self.conv1(x3)
        x3 = self.relu1(x3)
        x3 = self.conv2(x3)
        x3 = self.sig1(x3)
        x1_weight = x3[:, :128, :, :]
        x2_weight = x3[:, 128:, :, :]
        x3 = torch.mul(eeg, x1_weight) + torch.mul(eye, x2_weight)
        x3 = x3.reshape(x3.shape[0], -1)
        x3 = self.liner1(x3)

        return x3