# Face Recognition (FR) & Uncertainty Learning

> 汇报人：曹至欣
> 日期：2022.12.1

## 1. 人脸识别基础知识

### 1.1  人脸识别的分类

#### 1.1.1 按照任务分类

- 人脸验证（face verification）：1对1问题，即验证“你是不是某某人”
- 人脸识别（face identification / face recognition）：1对多问题，即验证“你是谁”

#### 1.1.2  按照测试样本的身份是否在训练集中分类

- 闭集问题（close-set）：通常意义上的分类数据集，训练集中的样本种类和测试集中的样本种类完全一致
- 开集问题（open-set）：训练集中的样本种类和测试集中的样本种类完全不同

### 1.2  开集FR和闭集FR问题

#### 1.2.1 储备知识

1. **度量学习**

   - 分类问题在实际应用中的困境：类别总数需要固定 & 训练集类别需要与实际应用中完全一致

   - 度量学习的目的是通过一个度量函数来衡量两个样本之间的相似程度

   - 为了使度量函数性能更好，通常需要将样本降维映射到一个低维空间，而这个映射关系，即是度量学习的模型

2. **FR中的三种数据集**

   - training set：用于模型的训练

   - gallery set：测试时使用，可理解为参考集。例如一个身份认证系统，每个注册ID在注册的时候都有一个照片，那么所有ID注册照片放在一起组成了gallery set

   - probe set：测试时使用，可理解为查询集。例如身份认证系统中，当用户下一次需要身份认证的时候，可能输入的是他当前状态的拍摄照片，这个照片同系统中他注册时候的照片进行比对，匹配成功则通过验证。那么所有ID身份认证时需要输进去比对的照片就组成了一个probe set

   > ***training set负责模型参数的调整，gallery set和probe set不负责模型参数的调整，将gallery set和probe set的数据放入模型中输出相应特征，并比对这两个特征,判断是gallery中的哪个人***

![image-20221124150346621](https://img-blog.csdn.net/20170309164649512?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMTU1NzIxMg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

#### 1.2.2 开集FR和闭集FR问题

1. 闭集FR问题
   - 人脸识别：基于softmax损失函数的多分类问题
   - 人脸验证：人脸识别的自然扩展，首先进行两次分类（一次针对测试图像，一次针对gallery），然后比较预测的身份是否相同
2. 开集FR问题
   - 人脸验证：通过两步来增大类间距离缩小类内距离的度量学习问题：首先训练一个特征提取器将人脸图像映射到判别性的特征空间，然后比较测试图像和gallery图像的特征向量距离是否超出一定阈值
   - 人脸识别：人脸验证的扩展，需要额外的步骤：比较测试图像和所有gallery图像的特征向量距离，然后根据最短的距离选择gallery的身份

![](https://pic4.zhimg.com/v2-7b79f8e92341b997c0daf8c5c5a8e8a7_r.jpg#pic_center)

## 2. 人脸识别论文分类

- 解决域和数据分布的问题
  - 例如训练域与测试域分布不同、数据集不平衡（存在种族偏差等问题）
  - 方法：域自适应、元学习等
- 解决模型压缩的问题
  - 即较高的计算资源消耗问题
  - 方法：知识蒸馏、FC层的优化等
- 细分问题
  - 例如年龄问题、化妆问题、侧脸问题等
  - 方法：注意力机制、多任务学习等
- 通用方法

## 3. 人脸识别中的不确定性学习

### 3.1 不确定性学习

- **确定性预测（deterministic prediction）**

  - 以人脸识别为例，例如人脸验证任务，使用人脸识别系统分别对两张照片提取特征，并用某种指标衡量两个特征的相似程度，假如相似程度很高则认为是一个人
  - 总结：通过预测一个确定性的人脸特征用来判断的方式被称为确定性预测

- **不确定性估计（uncertainty estimation）**

  - 确定性预测的弊端：当输入图像一张非常清晰，一张非常模糊，且人脸识别系统仍然得到较高的相似程度，是否认为结果可靠？

  - 置信度分数（confidence score）：用来判断机器给出的答案可信程度的多少

  - 不确定性：机器学习系统给出的判断不一定靠谱，即给出的判断具有一定程度的“不确定性”

  - 不确定性的分类：

    1. 数据的不确定性：数据中内在的噪声，即无法避免的误差，且不能通过增加采样数据来削弱（例如拍照手抖导致画面模糊，这张照片导致的不确定性不能通过增加拍照次数来消除）
       - 解决方案：提升数据采集的稳定性、提升衡量指标的精度来囊括各种客观因素

    2. 模型的不确定性：模型本身对输入数据的估计不准确，与某一单独的数据无关。可能导致的原因是训练不佳、训练数据不够等
       - 解决方案：具体问题具体分析，例如增加训练数据等

  ![image-20221130104944158](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221130104944158.png)
  
  

### 3.2 Probabilistic Face Embeddings（PFE）

> 第一个将data uncertainty learning运用到人脸识别任务中

**解决的问题**：解决数据的不确定性问题（模糊、遮挡、姿势等导致识别效果不佳）

> 实验验证不确定性对识别效果的影响：下图(a)表示同一ID，但是相似程度较低的图片，(b)是不同ID，但相似程度较高的图片

![image-20221201110448330](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221201110448330.png)

**核心思想**：用概率分布来替代传统方法学习到的确定性的人脸特征

![image-20221130112903230](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221130112903230.png)

> 传统方法：将输入人脸图像在潜在特征空间中映射成一个确定的点
>
> PFE引入不确定性，将输出向量认为是一个概率分布，不再是一个确定的点，它具有均值和方差，方差能够根据根据输入图像的质量、包含噪声的程度进行自适应的学习

$$
p(z_i|x_i) \sim N(z_i;\mu_i,\sigma_i)
$$

- $x_i$：第$i$个人脸样本
- $z_i$：第$i$个人脸样本对应的身份
- $$\mu_i$$：第$i$个样本的特征均值

- $$\sigma_i$$：第$i$个样本的方差（不确定性）

**具体实现方法**：给定一个预训练好的模型$f$，固定其参数不变化，认为embedding就是均值$\mu$，即$\mu(x) = f(x)$，然后在原有网络中加入的用来估计$\sigma(x)$的新的分支，该分支通过两个与瓶颈层共享相同输入的FC层实现 

**提出一种新的指标**：$mutual\space likelihood\space score\space (MLS)$用来衡量两个分布之间的距离
$$
s(x_i,x_j)=-\frac 12\sum_l^D\left(\frac {\left(\mu_i^{(l)}-\mu_j^{(l)} \right)^2}{\sigma_i^{2(l)}+\sigma_j^{2(l)}} +log(\sigma_i^{2(l)}+\sigma_j^{2(l)})\right)-const
$$

$$
const = \frac D2 log2\pi
$$



### 3.3 Data Uncertainty Learning in Face Recognition

**PFE方法的缺陷**：PFE仅仅学习了方差没有学习均值，即数据不确定性并没有真正用于影响模型中特征的学习 + PFE的评价指标MLS需要消耗更多运算时间和存储空间

**核心思想**：在PFE思路的引导下，既学习方差也学习均值，学习与身份相关的特征，且可使用传统的相似性度量方法不需要依赖MLS

![image-20221201092551632](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221201092551632.png)

> 能够预测概率分布，使得同一ID之间的样本均值距离拉进，而不同ID之间的样本均值距离拉开

**1. 基于分类的方法（从头学习一个分类模型）**：

![image-20221201092609486](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221201092609486.png)

> **为什么要引入$\epsilon$？
>
> 如果将每个样本看成一次采样自$N(z_i;\mu_i,\sigma_i^2I)$的随机嵌入$（p(z_i|x_i)=N(z_i;\mu_i,\sigma_i^2I)）$，而采样操作是不可微分，在模型训练过程中阻碍了梯度流的后向传播，所以从一个正态分布中随机采样一个噪声$\epsilon$

$$
s_i = \mu_i+\epsilon\sigma_i,\epsilon\sim N(0,I)
$$

**具体方法**：

这种方法在模型的backbone之后会有两条分支，分别用于预测均值$\mu$和方差$\sigma$。对于每一个样本的每一次迭代而言，都随机采样一个$\epsilon$ 。

通过这种方式得到的新样本特征$s_i$就是遵从均值$\mu$，方差为$\sigma$的高斯分布采出的值。通过这种简单的重新采样的技巧，就可以很好的进行模型训练。

**损失函数**：

包含softmax以及一切合理变种，还有一个KL损失，该损失项的引入来自于2016年一篇名为*Deep variational information bottleneck*的论文，能够使得每一个学出来特征的分布尽可能逼近单位高斯分布

> **为什么要引入KL损失？
>
> $\mu_i$都会被$\sigma_i$干扰，模型可能为了减少$s_i$梯度中不稳定的成分而预测一个小的$\sigma_i$，这样上式就变成了$s_i=\mu_i+c$，变成了确定性学习，所以引入KL损失用来限制$N(\mu_i,\sigma_i)$去接近一个正态分布$N(0,I)$

$$
L_{softmax}=\frac 1N\sum_i^N-log\frac{e^{w_{y_i}s_i}}{\sum_c^Ce^{w_cs_i}}
$$

$$
L_{kl}=KL[N(z_i|\mu_i,\sigma_i^2)||N(\epsilon|0,I)]=-\frac12(1+log\sigma_i-\mu_i^2-\sigma_i^2)
$$

$$
L_{cls}=L_{softmax}+\lambda L_{kl}
$$

**2. 基于回归的方法（从现有模型出发学习回归模型）**：

![image-20221201095237808](C:\Users\czx\AppData\Roaming\Typora\typora-user-images\image-20221201095237808.png)

> **为什么需要用分类层的权重矩阵$W$？
>
> 基于回归的方法需要连续映射空间$X\to Y$，而人脸数据集中$X$图像空间是连续的，但$Y$身份标签是离散的，所以需要构建新的且连续的目标映射空间，而$w_i$可认为是相同类别embedding的中心，所以被选为作为等价的映射空间

**需要回归的目标**：
$$
w_i=f(x_i)+n(x_i)
$$

- $f(x_i)$：理想的身份特征
- $n(x_i)$：样本$x_i$的不确定性信息或噪声

**具体方法**：

利用一个已经训练好的人脸识别模型，冻结其backbone，利用其分类层的权重矩阵$W$，其中$w \in R^{D×C}$，$w_i\in w$被认为是相同类别的embedding的中心。后续步骤与基于分类的DUL一致，然后新训练两条分支，一条用于预测均值，回归$w_c$的样本类中心；另一条用于预测方差。

**损失函数**：
$$
L_{rgs}=-\frac 1N\sum_i^N\left(\frac{(w_c-\mu_{i\in c})^2}{2\sigma_i^2}+\frac12ln\sigma_i^2+\frac12ln2\pi\right)
$$


## Reference:

[1] L. Tong et al., "FACESEC: A Fine-grained Robustness Evaluation Framework for Face Recognition Systems," 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021, pp. 13249-13258, doi: 10.1109/CVPR46437.2021.01305.

[2]Chang J , Lan Z , Cheng C , et al. Data Uncertainty Learning in Face Recognition[C]// 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020.

[3] Shi Y , Jain A . Probabilistic Face Embeddings[C]// 2019 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 2020.

[4] [https://zhuanlan.zhihu.com/p/76539587](https://zhuanlan.zhihu.com/p/76539587)

[5] [https://blog.csdn.net/u011557212/article/details/60963237](https://blog.csdn.net/u011557212/article/details/60963237)

[6] https://blog.csdn.net/Megvii_tech/article/details/103431546