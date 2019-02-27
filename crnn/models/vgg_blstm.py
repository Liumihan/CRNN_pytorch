import torch as t
from torch import nn

class VggBlstm(nn.Module):
    def __init__(self, image_size=(32, 280), num_classes=5990, max_label_length=10):
        super(VggBlstm, self).__init__()
        self.cnn = nn.Sequential()
        self.rnn = nn.Sequential()
        self.fc = nn.Sequential()

        self.cnn.add_module(name='conv1', module=nn.Conv2d(in_channels=1, out_channels=64,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN1', module=nn.BatchNorm2d(num_features=64))
        self.cnn.add_module(name='relu1', module=nn.ReLU())
        self.cnn.add_module(name='maxpl1', module=nn.MaxPool2d(kernel_size=(2, 2))) # 64 * 16 * 140 128

        self.cnn.add_module(name='conv2', module=nn.Conv2d(in_channels=64, out_channels=128,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN2', module=nn.BatchNorm2d(num_features=128))
        self.cnn.add_module(name='relu2', module=nn.ReLU())
        self.cnn.add_module(name='maxpl2', module=nn.MaxPool2d(kernel_size=(2, 2))) # 128 * 8 * 70 64

        self.cnn.add_module(name='conv3', module=nn.Conv2d(in_channels=128, out_channels=256,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN3', module=nn.BatchNorm2d(num_features=256))
        self.cnn.add_module(name='relu3', module=nn.ReLU())
        self.cnn.add_module(name='conv4', module=nn.Conv2d(in_channels=256, out_channels=256,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN4', module=nn.BatchNorm2d(num_features=256))
        self.cnn.add_module(name='relu4', module=nn.ReLU())
        self.cnn.add_module(name='maxpl3', module=nn.MaxPool2d(kernel_size=(2, 2))) # 256 * 4 * 35 32

        self.cnn.add_module(name='conv5', module=nn.Conv2d(in_channels=256, out_channels=512,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN5', module=nn.BatchNorm2d(num_features=512))
        self.cnn.add_module(name='relu5', module=nn.ReLU())
        self.cnn.add_module(name='conv6', module=nn.Conv2d(in_channels=512, out_channels=512,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN6', module=nn.BatchNorm2d(num_features=512))
        self.cnn.add_module(name='relu6', module=nn.ReLU())
        self.cnn.add_module(name='maxpl4', module=nn.MaxPool2d(kernel_size=(2, 2))) # 512 * 2 * 17

        self.cnn.add_module(name='conv7', module=nn.Conv2d(in_channels=512, out_channels=512,
                                                           kernel_size=(3, 3), stride=(1, 1),
                                                           padding=1))
        self.cnn.add_module(name='BN7', module=nn.BatchNorm2d(num_features=512))
        self.cnn.add_module(name='relu7', module=nn.ReLU())
        self.cnn.add_module(name='maxpl5', module=nn.MaxPool2d(kernel_size=(1, 2))) # 512 * 2 * 8

        # rnn_part
        self.rnn = nn.Sequential(
            Bilstm(in_channels=512, hidden_channels=256, out_channels=256),
            Bilstm(in_channels=256, hidden_channels=256, out_channels=512)
        )
        self.fc.add_module(name='pred', module=nn.Linear(in_features=512, out_features=5990))

    def forward(self, input):
        conv = self.cnn(input)
        B, C, H, W = conv.size()
        conv = conv.view(B, C, H*W)  # B C L
        conv = conv.permute(2, 0, 1)  # L B C
        seq = self.rnn(conv)
        out = self.fc(seq)
        return out


class Bilstm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Bilstm, self).__init__()
        self.rnn = nn.Sequential()
        self.rnn.add_module(name='bilstm',
                            module=nn.LSTM(input_size=in_channels, hidden_size=hidden_channels,
                                           bidirectional=True))
        self.embedding = nn.Linear(in_features=hidden_channels * 2, out_features=out_channels)

    def forward(self, *input):
        recurrent, _ = self.rnn(*input)
        seq_len, batch, hidden_size = recurrent.size()
        recurrent = recurrent.contiguous().view(seq_len*batch, hidden_size)
        output = self.embedding(recurrent)
        output = output.view(seq_len, batch, -1)

        return output



