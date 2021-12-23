# End2End Occluded Face Recognition by Masking Corrupted Features
> 汇报人：曹至欣  日期：2021-12-15

## 1.处理遮挡问题的任务分类

### 1.1 **OFR(occluded face recognition)**：

**目标：**将人脸图像与随机部分遮挡进行匹配

**与MFR的不同：**OFR遮挡位置不确定，目的是学习对于任何位置都可能出现的遮挡具有鲁棒性的特征；MFR遮挡位置已知，可以使用先验知识



### 1.2 **MFR(masked face recognition)**：

**提出背景：**疫情

**目标：**将戴口罩人脸图像与干净人脸图像进行匹配



### 1.3 **PFR(partial face recognition)**：

**目标：**将部分人脸图像与普通人脸图像进行匹配

**与MFR的不同（如果直接将PFR方法应用到MFR上):**

1. PFR需要一个预先定义的部分人脸作为输入，但这对MFR来说很难检测或者语义定义

2. PFR常常会忽略一些全局信息（例如下巴轮廓）

   

## 2.处理遮挡问题的方法分类

### 2.1 恢复

**思路：**指首先恢复遮挡的面部部分，然后对恢复的人脸图像进行识别。

**例子：**一种LSTM自动编码器模型，分为两个部分：第一部分多尺度空间LSTM编码器学习不同尺度的人脸表征，第二部分双通道结构LSTM解码器利用前一部分学到的表示进行人脸重建和遮挡检测。

**重难点：**保留身份信息的同时需要恢复面部部分



### 2.2 移除

**思路：**首先移除被遮挡破坏的特征，然后利用剩余的干净特征进行识别

**关键：**找到输入人脸图像的遮挡与深层特征之间的对应关系

**例子：**一种成对差分暹罗网络（PDSN），通过干净人脸和遮挡人脸学习mask 字典，处理随机遮挡时组合字典项得到特征丢弃mask与干净人脸相乘，移除掉被遮挡破坏的特征元素。

- 该例子的不足之处：需要额外的遮挡检测模块；训练时间长，训练规模大



## 3.FROM（face recognition with occlusion masks)

![image-20211216003848025](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216003848025.png)

### 3.1 feature pyramid extractor

**引入目的：**用于获得学习mask的空间感知特征+用于识别的可判别性特征

**结构：**应用LResnet50E-IR结构，作为模型的主干网络，该网络采用带有横向连接结构的自顶向下结构来构建金字塔特征

**输入：**未配对的随机遮挡和无遮挡的人脸图像

**输出：**金字塔特征X1、X2和X3，其中包含X3包含全局信息和局部信息，被送入mask decoder获得mask用于去除X1被破坏的元素

![image-20211216085732131](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216085732131.png)



### 3.2 mask decoder

**引入目的：**输入X3，获得mask用于去除X1被破坏的元素得到X1‘

**方法：**选取深层conv层的特征而不是fc层的特征

- 原因分析：需要得到不依赖于身份的只依赖于遮挡位置的mask，fc层保留了身份信息的同时丢失了空间信息

**结构：**![image-20211216090409471](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216090409471.png)

**生成的mask需要达到的目标：**

1. mask需要与输入图像的遮挡位置相关，通过occlusion pattern predictor实现
2. 正确消除掉那些由于遮挡而损坏的特征，通过CosFace Loss监督

**实验细节：**soft mask VS hard mask ？选择soft mask



### 3.3 occlusion pattern predictor

![image-20211216090847568](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216090847568.png)

**引入目的：**监督mask的学习，用于鼓励mask decoder生成与输入人脸的遮挡模式相关的mask

**启发思想：**一般相邻块在实际应用中具有相同的遮挡状态，eg.如果嘴被遮挡，鼻子很有可能也被遮挡

**实现思路：**将人脸分成K*K块，用不同大小的矩形覆盖相邻的块，右边的数值矩阵展示了每个不同大小遮挡矩阵能产生的遮挡模式的个数，其中（i，j)代表用i * j大小的矩阵覆盖人脸的遮挡模式的数量。最终我们能够获得![image-20211216092340419](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216092340419.png)种遮挡模式，比![image-20211216170827460](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216170827460.png)少了很多

**输入输出：**输入mask，输出occlusion feature vector

**结构：**![image-20211216092502021](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216092502021.png)

**备注：**能够很好的应用于干净人脸识别，即用0*0块的矩形去覆盖

**具体方法：**

1. 描述为分类问题(pattern prediction)

   - 在训练阶段，我们得到了每幅图像的遮挡位置。对于每个图像Xi，我们通过将其遮挡位置与226个(当K=5时)遮挡模式进行匹配来获得其遮挡模式yi。我们的匹配策略是计算遮挡模式和226个参考模式之间的IOU得分，然后选取IOU得分最大的模式作为对应的标签。

     ![image-20211216094532435](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216094532435.png)
     
     下面展示了输入图像和预测出来的遮挡模式：
     
     ![image-20211216093332214](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216093332214.png)

2. 描述为回归问题(pattern regression)

   - 不进行块划分的情况下直接回归遮挡区域的2D坐标，在训练阶段，我们对每一张人脸图像都有遮挡的矩形位置，并直接以左上角和右下角的二维坐标作为label，需要occlusion pattern predictor获得的vector

     ![image-20211216094440753](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216094440753.png)



### 3.4 loss

![image-20211216092707661](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216092707661.png)

其中识别人脸使用margin-based softmax loss(CosFace Loss，其中m1=m2=0):

![image-20211216105904694](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20211216105904694.png)



## reference:

Paper:

[1] Ding F ,  Peng P ,  Huang Y , et al. Masked Face Recognition with Latent Part Detection[C]// MM '20: The 28th ACM International Conference on Multimedia. ACM, 2020.

[2] Qiu H ,  Gong D ,  Li Z , et al. End2End Occluded Face Recognition by Masking Corrupted Features[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021, PP(99):1-1.

Blog：

[特征金字塔网络Feature Pyramid Networks - core! - 博客园 (cnblogs.com)](https://www.cnblogs.com/sdu20112013/p/11050746.html)

