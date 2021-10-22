# AC-CV组第二次组内分享 

> 汇报人：宋怡馨  2020-10-21
>

## Talking Head Generation

### 1. Definition 

给定一段驱动源（音频、视频、文字），可以根据任意一个人的面部特征与输入信息保持一致。

### 2. Key Points

* Identify Preserving 身份信息保留，也就是说生成的视频要尽可能地贴近输入的人脸图像。

* Visual Quality 视觉质量，e.g. 清晰度、相邻帧的平滑过渡

* Lip Synchronization 嘴唇动作与语音的同步

* Natural-spontaneous Motion 自然动作

### 3. Datasets

#### 3.1 without Head Movement

没有任何头部运动的数据集

1. GRID dataset (Cooke 2006)

   33 speakers, front-facing  the camera, 每个人说1000个短语，包括从51个词中随机选出的6个单词。所有的语句不带情感，没有任何明显的头部运动。

2. TCD-TIMID dataset  (Harte and Gillen 2015)

   62 speakers, 高质量音频和视频，总共读了6913个句子，没有明显的头部运动。角度包括正面和30°。

3. MODALITY dataset  (Czyzewski et al. 2017)

   Time-of-Flight camera，相机型号 SoftKinetic DepthSense 325，每秒 60 帧的速度提供深度数据，空间分辨率为 320 × 240 像素，可以检索3D数据。

4. LRW dataset (Chung and Zisserman 2016) 

   由上百个speakers说出的500个不同的词组组成，在这个数据集中，头部姿势存在显着差异——从一些视频中单个演讲者直接对着相机说话，到演讲者互相看着对方的小组辩论，导致一些视频具有极端的头部姿势。由于LRW是从现实世界中采集的，并带有真实标签（词），没有明显的头部运动。

#### 3.2 with Spontaneous Motions

在现实场景中人们说话时有自然的头部运动和情绪状态。下列数据集中，说话的人有适度和自然的头部运动。

1. CREMA-D dataset (Cao et al. 2014)  

   不同年龄组和种族的 91 位演员说出了 12 句话。 与其他数据集不同的是，CREMA-D 中的每个句子都是由演员以自然的头部运动以不同的情绪和强度多次表演的。

2. RAVDESS dataset (Livingstone et al. 2018) and MSP-IMPROV dataset (Busso et al. 2016)  

   涉及通过语音和面部表情传达的相互冲突的情感内容来创建刺激。

3. Faceforensics++ dataset (Rossler et al. 2019)   

   包含来自不同记者的 1000 个新闻发布会视频。 视频中的扬声器面向摄像机，头部运动适度自然。

4. ObamaSet (Suwajanakorn et al. 2017)  

   包含大量来自奥巴马总统每周总统演讲的视频片段，跨越八年。 当他说话时，他的头部姿势会发生变化，同时保持他的个性。 由于这些特征，ObamaSet 是一个合适的数据集，用于研究特定主题的高质量说话头生成。

#### 3.3 with Apparent Movement

上面两种数据集要么是在实验室控制的环境中录制的视频，要么是拍摄对象位于中心并面向摄像机的相对受控视频。以下是一些更具挑战性的数据集，其中包含明显头部运动或具有极端姿势的面部视频。

1. VoxCeleb1 (Nagrani et al. 2017) and VoxCeleb2 (Chung et al. 2018) datasets

   包含来自 6,000 多名演讲者的超过 100 万条话语，这些话语是从上传到 YouTube 的视频中提取的。 演讲者的口音、职业、种族和年龄各不相同。 数据集中包含的视频是在大量具有挑战性的视觉和听觉环境中拍摄的，光照、图像质量、姿势（包括配置文件）和运动模糊各不相同。其中包括面向大量观众的演讲，来自安静的室内工作室、室外体育场和红地毯的采访，专业拍摄的多媒体节选，甚至是在手持设备上拍摄的自制视频。 音频片段的质量会随着背景闲聊、笑声、重叠语音和不同的房间音响效果而下降。

2. The LRS3-TED dataset (Afouras et al. 2018)  

   用于视听语音识别任务的大规模数据集，其中包含字幕和音频信号之间的字级对齐。

3. MELD dataset (Poria et al. 2018)   

   包含来自电视剧《老友记》的 1,433 段对话中的约 13,000 条话语，并带有情感和情感标签。

### 4. Evaluation

* **Identify Preserving** 

  人脸检测：使用**ArcFace**提取的两幅图像特征之间的余弦距离ArcSim来度量两幅图像之间的身份相似性。最后取所有视频帧结果的avg。

* **Visual Quality** 

  用 **SSIM** 和 **FID** 来测量生成的视频帧与真实帧相比的感知错误和质量，其更好地模仿了人类的感知。 

  用**CPBD** 来基于合成视频帧中边缘的存在来确定模糊。

* **Lip Synchronization** 

  **Landmark Distance (LMD) **  计算合成视频帧和groundTruth之间唇部区域landmark的欧几里得距离。LMD估计唇形精度，可以表示合成视频帧和音频信号之间的唇形同步。

  **two-stream SyncNet ** 编码音频特征和视觉特征，然后使用对比损失来训练匹配问题。SyncNet训练中的false pairs只通过随机移位得到，所有的输出都与offset有关。可以在输入的音频和视频信号之间输出正确的同步误差（offset）。

* **Natural-spontaneous Motion** 

  User study

  

## Others

### 1. SSIM (Structural SIMilarity) 结构相似性

SSIM基于样本$x$和$y$之间的三个比较进行衡量：亮度 luminance、对比度 contrast、结构 structure。
$$
l(x,y)=\frac{2\mu_x\mu_y+c_1}{\mu_x^2+\mu_y^2+c_1} \ \ c(x,y)=\frac{2\sigma_x\sigma_y+c_2}{\sigma_x^2+\sigma_y^2+c_2} \ \ 
s(x,y)=\frac{\sigma_{xy}+c_3}{\sigma_x\sigma_y+c_3}
$$
$\mu$为均值、$\sigma^2$为方差、$\sigma_{xy}$为二者协方差。$c_1=(k_1L)^2,c_2=(k_2L)^2$，一般取$c_3=c_2/2$，常数的作用是避免除零。$L$为像素值的范围（如果像素值由$B$位二进制来表示 $L=2^B-1$），$k_1=0.01,k_2=0.03$。
$$
SSIM=[l(x,y)^{\alpha}\cdot c(x,y)^{\beta}\cdot s(x,y)^{\gamma}]
$$
设$\alpha=\beta=\gamma=1$，有：
$$
SSIM=\frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}
$$
每次计算都从图片上取一个$N\cross N$的窗口，然后不断滑动窗口进行计算，最后取平均值作为全局的 SSIM。

```python
ssim = skimage.measure.compare_ssim(im1, im2, data_range=255)
```



### 2. PSNR (Peak Signal-to-Noise Ratio) 峰值信噪比

* 灰度图计算法：

  给定一个$m\cross n$的参考图像 $I$ 和噪声图像 $K$，均方误差：
  $$
  MSE=\frac{1}{mn}\sum^{m-1}_{i=0}\sum^{n-1_{j=0}}[I(i,j)-K(i,j)]^2
  $$

  $$
  PSBR=10\cdot log_{10}(\frac{MAX^2_I}{MSE})
  $$

  其中$MAX^2_I$为图像最大像素值，一般uint8数据最大像素值为 255，浮点型数据最大像素值为 1。

* 彩色图像计算：
  - 分别计算 RGB 三个通道的 PSNR，然后取平均值。
  - 计算 RGB 三通道的 MSE ，然后再除以 3 。
  - 将图片转化为 YCbCr 格式，然后只计算 Y 分量也就是亮度分量的 PSNR。

```python
# method 1
diff = im1 - im2
mse = np.mean(np.square(diff))
psnr = 10 * np.log10(255 * 255 / mse)

# method 2
psnr = skimage.measure.compare_psnr(im1, im2, 255)
```





### 3. CPBD  cumulative probability blur detection 累积概率模糊检测

一种无参考的图像锐化定量指标评价因子。它是基于模糊检测的累积概率来进行定义，是基于分类的方法。CPBD的作者采用了高斯模糊的图像与JPEG压缩图像进行实验，表明CPBD是符合人类视觉特性的图像质量指标，值越大，所反映出的细节越清晰，模糊性越弱。因此，可以将此指标用于定量评判滤波后的图像的锐化质量。



### 4. FID Frechet Inception Distance score 弗雷切特起始距离

衡量生成图片的**质量**和**多样性**，用于评价不同的GAN模型。

inception network是一个特征提取的深度网络，最后一层是一个pooling层，然后可以输出一张图像的类别。

计算FID时，去掉这个最后一层pooling层，得到的是一个2048维的高层特征，以下简称n维特征。继续简化一下，这个n维特征是一个向量。则有：对于我们已经拥有的真实图像，这个向量是服从一个分布的，（我们可以假设它是服从一个高斯分布）；对于那些用GAN来生成的n维特征它也是一个分布；我们应该立马能够知道了，GAN的目标就是使得两个分布尽量相同。假如两个分布相同，那么生成图像的真实性和多样性就和训练数据相同了。

计算两个分布之间的距离：需要注意到这两个分布是多变量的，也就是前面提到的n维特征。也就是说计算的是两个多维变量分布之间的距离，数学上可以用Wasserstein-2 distance或者Frechet distance来进行计算。

假如一个随机变量服从高斯分布，这个分布可以用一个均值和方差来确定。那么两个分布只要均值和方差相同，则两个分布相同。我们就利用这个均值和方差来计算这两个单变量高斯分布之间的距离。但我们这里是多维的分布，我们知道协方差矩阵可以用来衡量两个维度之间的相关性。所以，我们使用均值和协方差矩阵来计算两个分布之间的距离。均值的维度就是前面n维特征的维度，也就是n维；协方差矩阵则是n*n的矩阵。

即多变量高斯分布之间的距离。

### 5. Ablation Study 消融研究

消融研究通常是指删除模型或算法的某些“功能”，并查看其如何影响性能。例如某论文提出了方法A,B,C。而该论文是基于某个baseline的改进。因此，在消融研究部分，会进行以下实验，baseline ，baseline+A，baseline+B, baseline+C, baseline+A+B+C等实验的各个评价指标有多少，从而得出每个部分所能发挥的作用有多大。可以理解为控制变量法。



## Reference

**Paper：**

​	[1] Chen, L., et al., What Comprises a Good Talking-Head Video Generation?: {{A Survey}} and {{Benchmark}}. arXiv:2005.03201 [cs, eess], 2020.

​	[2] Zhou, H., et al., Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation, in CVPR2021. 2021. p. 4176--4186.

​	[3] Ji, X., et al., Audio-Driven Emotional Video Portraits, in CVPR2021. 2021. p. 14080--14089.

**Blog：**

​	[1] https://www.cnblogs.com/LCarrey/p/14494489.html  

​	[2] https://zhuanlan.zhihu.com/p/50757421





