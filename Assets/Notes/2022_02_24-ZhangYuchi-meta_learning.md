# 元学习（meta learning）

> 主讲人：张与弛
> 日期：2022-02-23

## 1. 元学习

### 1.1 元学习定义

Meta-Learning: A Survey:
Meta-learning, or learning to learn, is the science of systematically observing how different machine learning approaches perform on a wide range of learning tasks, and then learning from this experience, or meta-data, to learn new tasks much faster than otherwise possible.

来自知乎的定义：
通常在机器学习里，我们会使用某个场景的大量数据来训练模型；然而当场景发生改变，模型就需要重新训练。但是对于人类而言，一个小朋友成长过程中会见过许多物体的照片，某一天，当Ta（第一次）仅仅看了几张狗的照片，就可以很好地对狗和其他物体进行区分。元学习Meta Learning，含义为学会学习，即learn to learn，就是带着这种对人类这种“学习能力”的期望诞生的。Meta Learning希望使得模型获取一种“学会学习”的能力，使其可以在获取已有“知识”的基础上快速学习新的任务，如:
- 让Alphago迅速学会下象棋
- 让一个猫咪图片分类器，迅速具有分类其他物体的能力


总结一下：

- 元学习的目标是让模型获得自主建模的能力
- 元学习通过多个任务训练模型

### 1.2 元学习的形式

#### 1.2.1 元学习与机器学习
在传统的机器学习中，我们通常需要手动设计网络结构，确定网络中的超参数，利用训练数据训练网络，得到一个贴合任务的神经网络模型<br>
在元学习中，我们希望设计一个元学习网络，该网络能够自动根据目标任务，生成对应的网络结构与超参数，从而实现自动化生成网络与自动化训练的功能<br>
二者的区别如下表所示：

||目的|输入|函数|输出|流程|
|:--|:--|:--|:--|:--|:--|
|机器学习|通过训练数据，学习到输入X与输出Y之间的映射，找到函数f|X|f|Y|1. 初始化f参数<br>2. 喂数据<X,Y><br>3. 计算loss，优化f参数<br>4. 得到y=f(x)|
|元学习|通过多个训练任务T及对应的训练数据D，找到函数F。F可以输出一个函数f，f可用于新的任务|训练任务及对应的训练数据|F|f|1. 初始化F参数<br>2. 喂训练任务T及对应的训练数据D，优化F参数<br>3. 得到f=F(*)<br>4. 新任务中y=f(x)|

### 1.3 MAML

#### 1.3.1 基础概念

MAML，全称为Model Agnostic Meta Learning，是元学习领域最为经典的算法之一。该算法的目的是获取一组更好的模型初始化参数，即让模型自己学会初始化。

#### 1.3.2 MAML的算法流程

1. 准备N个训练任务、每个训练任务对应的训练数据与测试数据。再准备几个测试任务，用于评估meta learning学习到的参数的效果。
2. 定义网络结构，如CNN。并初始化一个meta网络的参数为$\phi^0$，meta网络是最终用来应用到新的测试任务中的网络，该网络中存储了“先验知识”。
3. 开始执行迭代“预训练”:<br>
    a. 采样1个训练任务m。将meta网络的参数$\phi^0$赋值给任务m的网络，得到$\hat{\phi}^m$（初始的$\hat{\phi}^m$=$\phi^0$）<br>
    b. 使用任务m的训练数据，基于任务m的学习率$\alpha_m$，对$\hat{\phi}^m$进行1次优化，更新$\hat{\phi}^m$<br>
    c. 基于1次优化后的$\hat{\phi}^m$，使用任务m的测试数据，计算任务m的loss——$l^m(\hat{\phi}^m)$，并计算$l^m(\hat{\phi}^m)$对$\hat{\phi}^m$的梯度<br>
    d. 用该梯度，乘以meta网络的学习率$\alpha_meta$，更新$\phi^0$，得到$\phi^1$<br>
    e. 在训练任务上重复执行a-d，不断更新meta网络的参数$\phi$<br>
4. 通过3得到meta网络的参数$\phi$，该参数可以在测试任务中，使用测试任务的训练数据进行微调
5. 最终使用测试任务的测试数据评估meta learning的效果

#### 1.3.3 MAML与预训练的区别

从理念上：<br>
- 预训练的目的是在一个大的预训练集上学习到一组**性能较好**的网络参数，再将这组网络参数迁移到任务目标所在的数据集上，使网络在任务目标所在数据集上效果较好
- MAML的目的是在多个任务上学习到一组**学习潜力较高**的网络参数，再将这组网络参数迁移到目标任务上，通过微调促使网络能够快速收敛到一个效果较好的点上

从表达式上：<br>
- 预训练的损失函数可以表示为：$L(\phi)=\displaystyle \sum^{N}_{n=1}{l^n(\phi)}$
- MAML的损失函数可以表示为：$L(\phi)=\displaystyle \sum^{N}_{n=1}{l^n(\hat{\phi}^n)}$

#### 1.3.4 元学习的应用场景

- 小样本学习（few-shot learning）
- 领域自适应问题

### 1.4 其它的元学习算法

- Reptile
- Meta-RL

## 2. 人脸识别中的元学习

### 2.1 Learning Meta Face Recognition in Unseen Domains

#### 2.2.1 当前人脸识别的问题所在

人脸识别技术在实际的应用过程之中会遇到数据域不适应的问题。在训练时，模型往往仅针对种族、角度等单个领域进行训练，但在实际应用中，模型会遇到多种域问题，因此不能达到很好的效果。

#### 2.2.2 论文的想法

参照元学习的思想，利用训练集中不同的域对模型进行训练，提升模型在不同域下的适应能力，使模型在遇到未知域问题时能够保持较好的识别性能。

#### 2.2.3 论文的总体创新点

- 提出了广域人脸识别概念，强调了模型在未知域上的识别能力
- 提出了一种新的元人脸识别框架
- 设计了两个通用的人脸识别基准进行评估。对所提出的基准点进行的大量实验验证方法的有效性。


### 2.2 Cross-Domain Similarity Learning for Face Recognition in Unseen Domains

#### 2.2.1 论文的想法

本篇论文与上一篇论文想法相似，利用训练集上不同域之间的一致性概念，提出了跨域三重态损失（CDT），使得模型在不同域均能保持相似的性能。

#### 2.2.2 论文的总体创新点

- 提出了跨域三重态损失（CDT）
- 采用了元学习框架

# 3.参考资料

**Paper**：
[1] Vanschoren J. Meta-learning: A survey[J]. arXiv preprint arXiv:1810.03548, 2018.<br>
[2] Guo J, Zhu X, Zhao C, et al. Learning meta face recognition in unseen domains[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020: 6163-6172.<br>
[3] Faraki M, Yu X, Tsai Y H, et al. Cross-domain similarity learning for face recognition in unseen domains[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 15292-15301.<br>

**Blog**:
<https://zhuanlan.zhihu.com/p/136975128><br>
<https://zhuanlan.zhihu.com/p/289043310><br>
<https://zhuanlan.zhihu.com/p/108503451><br>
