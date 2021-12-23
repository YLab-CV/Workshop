---
author: Yuchi Zhang
author latest modified: Yuchi Zhang
date created: 2021-12-23, 20:22:00
date latest modified: 2021-12-23, 20:22:00
---

# 服务器使用规范

## 已安装

- Conda `Conda -V`：4.10.3
- Nvidia Driver `nvidia-smi`: 470.94
- System CUDA `nvcc -V`: 11.4
- Conda CUDA `conda list cudatoolkit`：11.3.1

## Conda 虚拟环境

- Base：Anaconda 官网默认环境
- PyTorch：PyTorch 官网默认环境
- 建议大家使用的时候使用以下命令创建属于自己或项目的新环境，尽量不要修改 Base 和 pytorch 环境！
	```shell
    # 新建空环境命令
    conda create --name [envname]

    # 复制环境命令
    conda create --name [new_env] --clone [old_env]

    # 例如
    conda create --name talkingface
    conda create --name talkingface --clone pytorch
    ```

## 硬盘情况

- 查看硬盘存储情况: 使用 `df -hl` 可以查看硬盘使用情况。
- `/DATA`：这个路径为 2T 机械硬盘挂载点
	- `/DATA/STORAGE`：存放用户数据，以后大家在使用服务器的时候使用 `mkdir` 命令以自己的名字创建一个属于自己的文件夹，将自己的数据和项目文件都放在这里，避免混乱和错删！
	- `/DATA/DATASETS`：这个路径用于存放开源数据集，请按照以下方式放置数据集，所有数据集必须包含 `README.md` ，描述以下内容：1. 数据集简介 2. 官网地址 3. 数据集论文 4. 数据集存储大小、时间、发布时间等
	```css
    └── DATA
        ├── STORAGE
        │    ├── User1
        │    ├── User2
        │    └── ...
        │    
        └── DATASETS
            ├── Dataset1
            ├── Dataset2
            ├── ...
            └── Dataset_name
                ├── README.md  # Dataset Info
                └── Dataset_name
                    ├── img1.jpg
                    ├── img2.jpg
                    ├── ...
                    └── imgn.jpg
    ```
