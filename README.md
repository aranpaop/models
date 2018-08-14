# models
Mainly for study.
Various base models used in CV problems.
Mainly built by keras.
Need improvement when using.
Keep updating.

alexnet:

1.使用11x11以及5x5卷积核，并且使用尺寸3x3步长2x2的最大池化。

2.使用relu激活函数。

3.由于relu不会像sigmoid以及tanh一样将数据集中到一个区域内，故使用了local response normalization。

4.使用了dropout。

![image](https://github.com/aranpaop/models/blob/master/alexnet.jpg)

lenet:

1.使用两个5x5卷积核。

![image](https://github.com/aranpaop/models/blob/master/lenet.jpg)

zfnet:

1.在alexnet的基础上使用7x7代替11x11卷积核。是对alexnet的优化。

![image](https://github.com/aranpaop/models/blob/master/zfnet.png)

vgg:

1.使用更小的3x3卷积核堆叠，并增加网络层数。

![image](https://github.com/aranpaop/models/blob/master/vgg.png)

resnet:

1.使用残差网络解决网络层数过深导致的准确率下降问题，原因是误差反向传播时由于梯度的逐层衰减而无法传递到较浅的网络。

2.如下图所示，将某一层的输出与前几层的输入相加，得到的结果输入下一层。要求长、宽、通道数均相等。右边是对左边的改进，使用1x1卷积核做线性映射减少通道数，使用3x3卷积核提取特征，再使用1x1卷积核增加到原来的通道数，代替左边的两个3x3卷积核，达到减少训练参数的目的。

![image](https://github.com/aranpaop/models/blob/master/resnetblock1.jpg)

3.如图中虚线所示。若模块输入与输出通道数不同，可以将输入在shortcut路径上进行1x1卷积改变通道数。若要将输入特征图的大小改变，可以在shortcut路径与主路径的第一层1x1卷积操作中改变卷积步长。

![image](https://github.com/aranpaop/models/blob/master/resnetblock2.png)

googlenet:

1.使用了inception网络，如左图，使用1x1、3x3、5x5以及3x3的maxpooling层进行多维度的特征提取并组合，为此作者给出了多种解释（特征丰富、稀疏矩阵分解成密集矩阵、赫布原理）。右图在3x3、5x5卷积之前以及pooling之后添加了一层1x1卷积（NIN），其作用是进行通道融合、减少参数等。

![image](https://github.com/aranpaop/models/blob/master/inception.png)
