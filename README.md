# CRNN_pytorch

**文本识别分为两部分：文本定位与文本序列识别。这个repo主要是做的后者。**

这是一个基于CRNN的文本序列识别项目。

在300w+的中文数据集上训练之后,得到了0.95的精度.(整个label都预测正确才认为正确)

我还做了一个基于keras的项目：https://github.com/Liumihan/CRNN_kreas， 个人认为keras对于新手来说更好上手，但是灵活性不够。所以自己又迁移到了pytorch上来。

#### File Description

| File                 | Description          |
| :------------------- | -------------------- |
| crnn/                | 模型相关             |
| crnn/data/part_300w  | 训练模型的数据集文件 |
| crnn/data/dataset.py | 数据集加载处理类     |
| crnn/models/crnn.py  | 模型文件             |
| crnn/trainer_weights | 训练好的权重文件     |
| crnn/config.py       | 配置文件             |
| crnn/utils.py        | 辅助函数             |
| train.py             | 训练模型程序         |
| evaluate.py          | 模型测试程序         |



#### 参考文献：

##### 论文：

CRNN：https://arxiv.org/abs/1507.05717

CTC：http://people.idsia.ch/~santiago/papers/icml2006.pdf

##### 博客：

CRNN：

https://zhuanlan.zhihu.com/p/43534801

CTC：

https://www.cnblogs.com/qcloud1001/p/9041218.html，

https://distill.pub/2017/ctc/

https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c

训练数据集：

链接: https://pan.baidu.com/s/1MinLf7IJvIAKK80wWJWPKg 提取码: yjjn 

##### git：

https://github.com/Liumihan/CRNN-Keras

https://github.com/Liumihan/keras_ocr