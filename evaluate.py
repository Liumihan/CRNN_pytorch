from crnn.config import opt
from crnn.data.dataset import TextDataset, ToTensor, ZeroMean, Rescale, Gray
from matplotlib import pyplot as plt
import torch
from crnn.data.dataset import CharClasses
from torchvision import transforms
from crnn.utils import ctc_decode, show_image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

def check_acc(test_dataset, crnn_model):
    '''
    :param test_dataset: ~Dateset 需要验证的数据集
    :param crnn_model:  ~nn.Sequential 模型数据
    :return: running_acc ~float 返回正确率
    整个标签一字不差都正确才认为是正确的标签
    '''
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    counter = 0
    running_acc = 0.0
    for batch in tqdm(test_dataloader):
        images, labels, label_length = batch['image'], batch['label'], batch['label_length']
        preds = crnn_model(images.to('cuda:0'))
        pred_labels = ctc_decode(preds)
        for p, gt, l in zip(pred_labels, labels.numpy(), label_length.numpy()):
            if p == gt[:int(l)].tolist():
                correct = 1
            else:
                correct = 0
            running_acc  = (running_acc*counter + correct) / (counter + 1)
            counter += 1
    return running_acc


if __name__ == '__main__':
    char_dict = CharClasses(opt.char_dict_file).chars
    crnn_model = torch.load(opt.load_path)
    crnn_model.eval()  # 记住一定要使用这个
    test_dataset = TextDataset(txt_file=opt.val_filename, root_dir=opt.root_dir,
                               max_label_length=10, transform=transforms.Compose([Rescale((32, 280)), Gray(), ZeroMean(), ToTensor()]))
    acc = check_acc(test_dataset, crnn_model)
    print(acc)
    # data = test_dataset[10]
    # image, gt_label, gt_lable_length = data['image'], data['label'], data['label_length']
    # show_image(data, char_dict)
    # pred = crnn_model(image[None, :, :, :].to('cuda:0'))
    # label = ctc_decode(pred=pred)
    # text = [char_dict[num] for num in label[0]]
    # print(text)
pass