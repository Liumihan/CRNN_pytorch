import torch
from torch import nn
import torch.nn.functional as F

device = torch.device('cuda:0')


# 我的图片是H, W = 32, 280
class CRNN(nn.Module):
    def __init__(self, pic_channels=1, rnn_hidden_channels=256, nclasses=5990):
        super(CRNN, self).__init__()
        kernels = [3, 3, 3, 3, 3, 3, 3]
        padding = [1, 1, 1, 1, 1, 1, 1]
        strides = [1, 1, 1, 1, 1, 1, 1]
        channels = [64, 128, 256, 256, 512, 512, 512]

        self.cnn = nn.Sequential()
        self.rnn_hidden_channels = rnn_hidden_channels

        def convRelu(i, BN=False):
            if i == 0:
                in_channel = pic_channels
            else:
                in_channel = channels[i-1]
            out_channel = channels[i]
            self.cnn.add_module(name='conv{0}'.format(i),
                                module=nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                                 kernel_size=kernels[i], padding=padding[i],
                                                 stride=strides[i]))
            if BN:
                self.cnn.add_module(name='batch_norm{0}'.format(i), module=nn.BatchNorm2d(num_features=out_channel))
            self.cnn.add_module(name='relu{0}'.format(i), module=nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        self.cnn.add_module(name='maxpool1', module=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        convRelu(2)
        convRelu(3)
        self.cnn.add_module(name='maxpool2', module=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        convRelu(4, BN=True)
        self.cnn.add_module(name='maxpool3', module=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
        convRelu(5, BN=True)
        self.cnn.add_module(name='maxpool4', module=nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        convRelu(6, BN=True)
        self.cnn.add_module(name='maxpool5', module=nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))
        if self.rnn_hidden_channels:
            self.rnn = nn.Sequential(
                Bilstm(in_channels=512, hidden_channels=self.rnn_hidden_channels, out_channels=self.rnn_hidden_channels),
                Bilstm(in_channels=self.rnn_hidden_channels, hidden_channels=self.rnn_hidden_channels, out_channels=nclasses)
            )
        else:
            self.fc = nn.Linear(in_features=512, out_features=nclasses)

    def forward(self, *input):
        conv = self.cnn(*input)
        b, c, h, w = conv.size()
        conv = conv.view(b, c*h, w)
        conv = conv.permute(2, 0, 1)
        if self.rnn_hidden_channels is not None:
            out = self.rnn(conv)
        else:
            out = self.fc(conv)

        # T, N, C = out.size()
        # out = out.view(T, N*C)

        return out


class Bilstm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, BN=True):
        super(Bilstm, self).__init__()
        self.BN = BN
        self.rnn = nn.Sequential()
        self.rnn.add_module(name='bilstm',
                            module=nn.LSTM(input_size=in_channels, hidden_size=hidden_channels,
                                           bidirectional=True))
        if BN:
            self.BN = nn.BatchNorm1d(num_features=int(hidden_channels * 2))
        self.embedding = nn.Linear(in_features=hidden_channels * 2, out_features=out_channels)

    def forward(self, *input):
        recurrent, _ = self.rnn(*input)
        if self.BN:
            recurrent = recurrent.permute(0, 2, 1)
            recurrent = self.BN(recurrent)
            recurrent = recurrent.permute(0, 2, 1)
        seq_len, batch, hidden_size = recurrent.size()
        recurrent = recurrent.contiguous().view(seq_len*batch, hidden_size)

        output = self.embedding(recurrent)
        output = output.view(seq_len, batch, -1)

        return output


if __name__ == '__main__':
    crnn = CRNN(pic_channels=1, rnn_hidden_channels=256, nclasses=5990)
    print(crnn)
    # # Note pytorch的 conv网络的input的类型只能是float类型的
    # input = torch.randint(low=0, high=255, size=(1, 1, 32, 280), dtype=torch.float32)
    # out = crnn(input)
    # out = out.detach().numpy()
    # print(out.shape)
