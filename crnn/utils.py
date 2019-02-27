import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt


def ctc_decode(pred, blank_index=0):  # T * N * C
    arg_max = pred.argmax(dim=-1)  # T * N
    arg_max = arg_max.t()  # N * T
    arg_max = arg_max.to(device='cpu').numpy()
    pred_labels = []
    for line in arg_max:
        label = [k for k, g in itertools.groupby(line)]
        while blank_index in label:
            label.remove(blank_index)
        pred_labels.append(label)
    return pred_labels  # type: list


def show_image(sample, char_dict):
    '''
    :param sample: Testdata经过了transform的返回的一个sample
    :param char_dict:
    :return:
    '''
    image, label, label_length = sample['image'], sample['label'], sample['label_length']
    image = (image + 1.0) / 2.0 * 255.0
    image = image.numpy()
    image = np.squeeze(image, axis=0)
    text = [char_dict[num] for num in label.numpy().astype(np.int32)]
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    plt.imshow(image.astype(np.uint8), cmap='gray')
    ax.set_title(str(text))
    plt.show()



if __name__ == '__main__':
    pred = torch.tensor([[1, 2, 3], [1, 1, 1]], dtype=torch.float)
    target = torch.tensor([[1, 2, 3], [2, 1, 2]], dtype=torch.float)
