from torch.utils.data import DataLoader
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray, RandomConvert
from torch.nn import CTCLoss, init
from torch import optim
from crnn.models.crnn import CRNN
from crnn.utils import ctc_decode
import torch
from crnn.config import opt
from torchvision import transforms
import time
import warnings
import os
from tqdm import tqdm, trange
import gc
'''
训练网络程序：
每次一个epoch查看running_loss, 如果loss小于之前的loss则替换
'''
warnings.filterwarnings("ignore")


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out')
        init.constant_(m.bias, 0)


train_dataset = TextDataset(txt_file=opt.train_filename, root_dir=opt.root_dir,
                            transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]),
                            max_label_length=opt.max_label_length)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

device = opt.device
# 是否继续训练
if opt.load_path:
    net = torch.load(opt.load_path)
else:
    net = CRNN()

net.apply(weights_init)
print(net)
print(net.parameters())
net = net.to(device=device)
net.zero_grad()

params = net.parameters()

ctc_loss = CTCLoss(blank=0)
optimizer = optim.Adam(params=params, lr=0.001, weight_decay=1e-5)
best_loss = 50
print('gc is enabel:', gc.isenabled())
for epoch in trange(opt.epoch):
    running_loss = 0.0
    for i, train_data in tqdm(enumerate(train_loader, 0)):
        inputs, labels, labels_length = train_data['image'], train_data['label'], train_data['label_length']

        preds = net(inputs.to(device))
        optimizer.zero_grad()  # 重点pytorch中必须要有这一步
        pred_labels = ctc_decode(preds)

        log_preds = preds.log_softmax(dim=2)
        targets = labels.to(device=device, dtype=torch.float32)
        input_lengths = torch.tensor([len(l) for l in preds.permute(1, 0, 2)], dtype=torch.float32, device=device)
        target_lengths = torch.tensor(labels_length, device=device, dtype=torch.float32)

        loss = ctc_loss(log_preds, targets, input_lengths, target_lengths)
        running_loss += loss.item() * len(train_data)
        print('epoch:{}, iter:{}, loss:{}'.format(epoch, i, loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=params, max_norm=0.1)  # 防止梯度爆炸
        optimizer.step()
    epoch_loss = running_loss / len(train_dataset)
    print('epoch:{}, epoch loss:{}'.format(epoch, epoch_loss))

    if epoch_loss < best_loss:
        weights_dir = './crnn/trained_weights/'
        for file in os.listdir(weights_dir):
            os.remove(os.path.join(weights_dir, file))
        current = time.asctime(time.localtime(time.time()))
        torch.save(net, f='./crnn/trained_weights/epoch_{}_epoch_loss{:.5f}_time_{}.pt'.format(epoch, epoch_loss, current))


