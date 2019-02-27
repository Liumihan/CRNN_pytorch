from pprint import pprint
class Config:
    # data

    # 300w
    train_filename = '/home/liumihan/Desktop/OCR/CRNN/data/img_300w_and_trainedweight/txt/train.txt'
    val_filename = '/home/liumihan/Desktop/OCR/CRNN/data/img_300w_and_trainedweight/txt/test.txt'
    root_dir = '/home/liumihan/Desktop/OCR/CRNN/data/img_300w_and_trainedweight/img'
    # part 300w
    # train_filename = '/home/liumihan/Desktop/OCR/CRNN/data/part_300w/txt/train.txt'
    # val_filename = '/home/liumihan/Desktop/OCR/CRNN/data/part_300w/txt/test.txt'
    # root_dir = '/home/liumihan/Desktop/OCR/CRNN/data/part_300w/img'

    char_dict_file = '/home/liumihan/PycharmProjects/ocr/crnn/data/part_300w/txt/m_char_std_5990.txt'
    image_size = (32, 280)
    max_label_length = 10

    # cuda
    device = 'cuda:0'

    # network
    nclasses = 5990

    # training
    epoch = 100

    # model
    load_path = "/home/liumihan/PycharmProjects/ocr/crnn/trained_weights/epoch_2_epoch_loss0.00045_time_Wed Feb 27 10:13:23 2019.pt"
    trained_weights = '/home/liumihan/PycharmProjects/ocr/crnn/epoch_0_epoch_loss0.02102_time_Tue Feb 26 21:45:24 2019.pt'
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('Unknow Option: "--%s"' % k)
            setattr(self, k, v)
        print('**********************************user config*************************')
        pprint(self._state_dict())
        print('*************************************end******************************')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() if not k.startswith('_')}

opt = Config()
