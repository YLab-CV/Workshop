---
creator: Guanyu Hu
date created: 2021-10-26, Tuesday, 19:15:40
date modified: 2021-11-02, Tuesday, 11:45:27
latest modified author: Guanyu Hu
---

# Generative Adversarial Network

> 主讲人：胡冠宇
> 日期：2021-11-01

## 1. Generative Model

### 1.1. What Are Generative Models

#### 1.1.1. Discriminative Models - 判别模型

discriminative [/dɪ'skrɪməˌneɪtɪv/](https://dictionary.blob.core.chinacloudapi.cn/media/audio/tom/0d/e3/0DE330601451C45D5EABD53E04B71672.mp3) adj. 歧视的；有辨别力的；有判别力的

- **Purpose**:
	- Discriminative models 主要用于分类，也被称为 classifiers(分类器)
- **How:**
	- 直接根据特征 $X$ ，来对 $Y$建模，**划定一个整体判别边界**，每新来一个数据 $x$，就根据这个边界来判断它应该属于哪一类。[^1]
	- 获得样本 $X$ 属于类别 $Y$ 的概率分布，是一个条件概率 $P(Y|X)$
- **Goal:**
	- 给定一组 feature $X$ 来判定它是属于哪个分类 $Y$ 的

#### 1.1.2. Generative Models - 生成模型

- generative [/ˈdʒen(ə)rətɪv/](https://dictionary.blob.core.chinacloudapi.cn/media/audio/tom/2f/d4/2FD4A110F56A6383BF3FC4A8C971BB8E.mp3) adj. 有生产力的；能生产的；生成性；生成的
- **Purpose**: 主要用于分类和生成

##### 1.1.2.1. For Classification

- **How**:
	- 观察训练数据 $X$与$Y$的整体分布，求得联合概率分布 $P(X,Y)$，每新来一个数据 $X$，求出 $X$与不同分类$Y$之间的联合概率分布，将 $X$分为联合概率大的那一类。
- **Goal**:
	- 给定一组 feature $X$ 计算出与不同分类的联合概率 $P(X,Y)$
- **Methods**：[^2]
	- 朴素贝叶斯方法：我们通过数据集学习到先验概率分布$P(Y)$和条件概率分布$P(X|Y)$，即可得到联合概率分布$P(X,Y)$； $P(X,Y) = P(X|Y) \cdot P(Y)$
	- 隐马尔可夫模型：我们通过数据集学习到初始概率分布、状态转移概率矩阵和观测概率矩阵，即得到了一个可以表示状态序列和观测序列的联合分布的马尔可夫模型。

##### 1.1.2.2. For Generation

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101174251.png)

- **Purpose**:
	- Try to learn how to make a realistic representation of some class
- **How**:
	- Take some random input represented by the noise and sometimes takes in a class $Y$(condition)
	- From these inputs, to generate a set of features $X$ that look like a realistic representation of class $Y$.
- **Methods**：
	- VAE: 近似得到 $P(X)$ 的概率密度函数
	- GAN: 直接产生符合 $X$ 本质分布的样本

#### 1.1.3. Discriminative Models VS. Generative Models

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211028015315.png)

- **例子[^3]**： 确定一个羊是山羊还是绵羊
	- 判别式模型举例：是从历史数据中学习到模型，然后通过提取这只羊的特征来预测出这只羊是山羊的概率，是绵羊的概率。
	- 生成式模型举例：根据山羊的特征学习出一个山羊的模型，然后根据绵羊的特征学习出一个绵羊的模型，从这只羊中提取特征，放到山羊模型中看概率是多少，在放到绵羊模型中看概率是多少，哪个大就是哪个。

### 1.2. Types of Deep Generative Models.

#### 1.2.1. Variational Autoencoders - VAE(变分自编码器)

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101180324.png)

- **What:**
	- variational 英: [/veərɪ'eɪʃnəl/](https://dictionary.blob.core.chinacloudapi.cn/media/audio/tom/4d/9b/4D9B6753A36E9636DA3B6265AB7E322E.mp3) adj. 变化；〔数〕变分；变量的；变异的
	- Composed with two models: an encoder and a decoder, these are typically neural networks.
- **How:**
	1. They learn first by feeding in realistic images into the encoder, then the encoder's job is to find a good way of representing that image in the Latent Space
	2. Take the latent representation and put it through the decoder, it reconstruct the realistic image that the encoder saw before
	3. 在训练完成后 去掉 encoder, 我们可以从隐空间中随机选择一个点，decoder 就可以生成逼真的图片
		![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101180848.png)
- **Variational**：
	- 给整个模型注入一些噪音, encoder 没有将图片在隐空间中编码为一个点, 实际上是**将图像编码到整个分布上**，然后对该分布上的一个点进行采样，以输入 decoder 从而生成一个真实的图像。
		![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211031011334.png)

#### 1.2.2. Generative Adversarial Networks - GAN(生成对抗网络)

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211028155440.png)

- **What:**
	- adversarial [/ˌædvɜrˈseriəl/](https://dictionary.blob.core.chinacloudapi.cn/media/audio/tom/25/c5/25C5E97B7C56BDE31279F4D8189E95A1.mp3) adj. 对立的；敌对的
	- Composed with two models: a discriminator and a generator , these are typically neural networks .

- **How:**
	- Generator takes a noise vector and an optional class as input. As an output, it can generate a realistic representation of the class.
	- Discriminator looking at fake and real representation and simultaneously trying to figure out which ones are real and which ones are fake.
	- Over time, each model tries to one up(胜人一筹) each other, compete against each other, this is where the term "adversarial" comes from.

- **Difference with VAE**:
	- There's no guiding encoder this time that determines what input noise vector should look like
	- GAN 是通过判别器网络来进行优化的，让生成器产生数据的分布直接拟合训练数据的分布，而对具体的分布没有特别的要求。

## 2. Generative Adversarial Networks - GAN

### 2.1. Discriminator

- **Discriminator 对自己的影响**:
	- improve itself
		![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101183958.png)

- **Discriminiator 对 Generator 的影响**:
	- improve generator
		![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101184607.png)

	- This $0.85$ will be given to the generator to improve its efforts

- **Summary**:
	- The discriminator is a classifier
	- It learns the probability of class $Y$(real or fake) given features $X$
	- The probabilities are the feedback for the generator

### 2.2. Generator

#### 2.2.1. What the Generator Does

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211027173859.png)

- The final goal is to generate a realistic representation of a certain class.
- 这里神经网络的输出不是一个分类而是图片中的每个像素点
- 为了使每次生成的图像都不同，每一次 run 需要输入 different sets of random values - noise vector.

### 2.3. BCE Cost Function

- BCE: Binary Cross Entropy function

$$
J(\theta)=-\frac{1}{m} \sum_{i=1}^{m}\left[y^{(i)} \log h\left(x^{(i)}, \theta\right)+\left(1-y^{(i)}\right) \log \left(1-h\left(x^{(i)}, \theta\right)\right)\right]
$$

- 公式：
	- 取一个 batch 中所有 cost 的平均值
	- $h$: predictions made by the model
	- $y$: is the labels for the different examples, true label of real fake
	- $x$: feature passed in through the prediction, could be an image
	- $\theta$: parameters to model the classifier $h$
- 理解：
	- Close to zero when the label and the prediction are similar
	- Approaches infinity when the label and the prediction are defferent

### 2.4. Training GANs

#### 2.4.1. Training Discriminator

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211028163739.png)

#### 2.4.2. Training Generator

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211028163810.png)

1. First, you have a noise vector $\xi$ [/saɪ/], pass into a generator to produce a set of features that compose an image.
2. This image $\hat{X}$ is feed into the discriminator, which determines how real and how fake it is, and output a $\hat{Y_d}$
3. Compute a cost function 去计算 discriminator 将生成的图片识别为 real 的距离并根据距离和梯度方向更新 generator 的参数
	- 因为 generator 希望生成的图片尽可能真实，即让$\hat{Y_d}=1$
	- 而 discriminator 希望尽可能识别出这是生成的图片，即让$\hat{Y_d}=0$
4. 当 generator 效果满足要求的时候就可以将模型保存，it can generate all sorts of different examples

![](imgs/2021_11_01-GuanyuHu-Generative_Adversarial_Network/Pasted%20image%2020211101191626.png)

- **Question:**
	- Why need noise?
	- Noise is a random distribution vector
	- $Y$ with the added noise as input, model would generate 可信且多样化的 $Y$
- 参考：
	- [Why Do GANs Need So Much Noise?](https://towardsdatascience.com/why-do-gans-need-so-much-noise-1eae6c0fb177)
	- [对抗生成网络（GAN）为什么输入随机噪声？](https://www.zhihu.com/question/320465400/answer/1051141660)

**原理**：生成器所做的是学习从一些潜在空间$\xi$(输入)到一些采样空间 $\hat{X}$ (输出)的映射，并理解学习是如何进行的。

#### 2.4.3. Training GANs

- Both models should improve together and should kept at similar skill levels from the beginning of training
- 如果 discriminator 特别好，那么会将所有生成的都预测为 fake，这样 generator 就不知道如何改进
- 如果 generator 特别好，那么 discriminator 则会将所有生成的图像识别为 real，同样 generator 也不知道如何改进
- 只有两种模型一起改进，并保持在同一个 level 上面。
- 常见的问题是 discriminator 会学习的快，因为只需要区分 fake and real，而 generator 需要 model the entire space，discriminator 的任务比 generator 的容易很多，

[^1]: [机器学习“判定模型”和“生成模型”有什么区别？ - 知乎](https://www.zhihu.com/question/20446337)
[^2]: [GAN 的优化（一）生成模型与 GAN - 知乎](https://zhuanlan.zhihu.com/p/68961138)
[^3]: [机器学习中判别模型和生成模型 - 知乎](https://zhuanlan.zhihu.com/p/261119069)
