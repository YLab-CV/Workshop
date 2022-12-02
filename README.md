---
date created: 2021-10-15, Friday, 13:58:57
date modified: 2021-11-23, Tuesday, 21:03:19
latest modified author: Yixin Song
date latest modified: 2022-11-04,11:23:45
---

# YLab-CV-Workshop

## Notice

- Workshop 需要准备的内容为 **Note** 和 **Slide**
- 文件格式：
	- Slide：`PPTX` + `PDF`
	- Note：`MD`
- 命名规则： 日期-姓名-文件名
	- 日期、姓名、文件名之间以 `-` 连接
	- 其他以 `_` 连接
	- 不要有空格
	- 例如：`2021_10_15-JieWei-Domain_Adaptation`
- 文件位置：
	- Slide 位置：`Assets/Slides/`
	- Note 位置：`Assets/Notes/`
		- Note 中如果有图片则将图片打包到文件夹中，文件夹以 Note 名称命名：`Assets/Notes/imgs/pre_time-YourName-Pre_Topic`
		- 例如 Note 名称为 `2021_10_14-JieWei-Domain_Adaptation.md`，则图片打包文件夹名称为 `2021_10_14-JieWei-Domain_Adaptation`
	- 其他文件放入：`Assets/Files`
		- Code 位置：`Assets/Files/Codes`, 命名方式如上

【其他】：

- GIthub 默认没有对 Markdown 中的 LaTeX 语法进行支持，可以安装 Chrome 插件解决： [TeX All the Things](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn?hl=en)

## Workshop Weekly

| Date       | Week   | Topic                                           | Presenters | Slides                                                                                                                                                                  | Notes                                                                                                                                                    | Appendix                                                                         |   |
| ---------- | ------ | ----------------------------------------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | - |
| 2021-10-14 | Week01 | Domain Adaptation                               | 魏 洁        | [Slide](Assets/Slides/2021_10_14-JieWei-Domain_Adaptation.pptx) | [Note](Assets/Notes/2021_10_14-JieWei-Domain_Adaptation.md) | -                                                                                |   |
| 2021-10-14 | Week01 | Visual Understanding Overview                   | 张硕、曹至欣     | -                                                                                                                                                                       | [Note](Assets/Notes/2021_10_14-ZhangShuo_Cao-Visual_Understanding_Overview.md) | -                                                                                |   |
| 2021-10-21 | Week02 | Talking-head Video Generation: A Survey         | 宋怡馨        | [Slide](Assets/Slides/2021_10_21-YixinSong-Talking_Head_Generation.pptx) | [Note](Assets/Notes/2021_10_21-YixinSong-Talking_Head_Generation.md) | -                                                                                |   |
| 2021-10-21 | Week02 | Object Detection: A Survey                      | 曹至欣        | [Slide ](Assets/Slides/2021_10_21-ZhixinCao-Object_Detection_A_Survey.pptx) | [Note](Assets/Notes/2021_10_21-ZhixinCao-Object_Detection_A_survey.md) | -                                                                                |   |
| 2021-11-01 | Week03 | Generative Adversarial Network (GAN)            | 胡冠宇        | [Slide](Assets/Slides/2021_11_01-GuanyuHu-Generative_Adversarial_Network.pptx) / [PDF](Assets/Slides/Slides_PDF/2021_10_28-GuanyuHu-Generative_Adversarial_Network.pdf) | [Note](Assets/Notes/2021_11_01-GuanyuHu-Generative_Adversarial_Network.md) | [Code](Assets/Files/Codes/2021_11_01-GuanyuHu-Generative_Adversarial_Network.py) |   |
| 2021-11-01 | Week03 | Semantic Segmentation & Image Caption: A Survey | 张 硕        | [Slide](Assets/Slides/2021_11_01-ZhangShuo-Panoptic_Segmentation_survey.pptx) | [Note](Assets/Notes/2021_11_1-ZhangShuo-Panoptic_Segmentation_survey.md) | -                                                                                |   |
| 2021-11-04 | Week04 | Metric Learning                                 | 张与弛        | [Slide](Assets/Slides/2021_11_04-YuchiZhang-metric_learning.pptx) / [PDF](Assets/Slides/Slides_PDF/2021_11_04-YuchiZhang-metric_learning.pdf) | [Note](Assets/Notes/2021_11_04-YuchiZhang-metric_learning.md) | -                                                                                |   |
| 2021-11-15 | Week05 | Sequence Model                                  | 魏 洁        | [Slide](Assets/Slides/2021_11_15-JieWei-Sequence_Model.pptx) / [PDF](Assets/Slides/Slides_PDF/2021_11_15-JieWei-Sequence_Model.pdf) | [Note](Assets/Notes/2021_11_15-JieWei-Sequence_Model.md) | -                                                                                |   |
| 2021-11-22 | Week06 | Convolution Arithmetic                          | 胡冠宇        | -                                                                                                                                                                       | [Note](Assets/Notes/2021_11_22-GuanyuHu-Convolution_Arithmetic.md) / [PDF](Assets/Notes/Notes_PDF/2021_11_22-GuanyuHu-Convolution_Arithmetic.pdf) | [Code](Assets/Files/Codes/2021_11_22-GuanyuHu-Convolution_Arithmetic.py) | - |
| 2021-12-03 | Week07 | ViT & Masked Autoencoders                       | 张硕         | [Slide](Assets/Slides/2021_12_3-ZhangShuo-ViT_MAE.pptx) | [Note](Assets/Notes/2021_12_3-ZhangShuo-ViT_MAE.md) / [PDF](Assets/Notes/Notes_PDF/2021_12_3-ZhangShuo-ViT_MAE.pdf) | -                                                                                |   |
| 2021-12-03 | Week08 | Talking Head Paper Sharing                      | 宋怡馨        | [Slide](Assets/Slides/2021_12_13-YixinSong-Talking_Head_Generation.pptx) | -                                                                                                                                                        | -                                                                                |   |
| 2021-12-16 | Week09 | Occluded Face Recogniton                        | 曹至欣        | [Slide ](Assets/Slides/2021_12_16-ZhixinCao-Occluded_Face_Recognition.pptx) | [Note](Assets/Notes/2021_12_16-ZhixinCao-Occluded_Face_Recogniton.md) / [PDF](Assets/Notes/Notes_PDF/2021_12_16-ZhixinCao-Occluded_Face_Recognition.pdf) | -                                                                                |   |
| 2022-01-06 | Week10 | Audio Signal Processing Part_1                  | 胡冠宇        | -                                                                                                                                                                       | [Note](Assets/Notes/2022_01_06-GuanyuHu-Audio_Signal_Processing.md) / [PDF](Assets/Notes/Notes_PDF/2022_01_06-GuanyuHu-Audio_Signal_Processing.pdf) | [Code](Assets/Files/Codes/2022_01_06-GuanyuHu-Audio_Signal_Processing.7z) |   |
| 2022-01-13 | Week11 | Audio Signal Processing Part_2                  | 胡冠宇        | -                                                                                                                                                                       | [Note](Assets/Notes/2022_01_06-GuanyuHu-Audio_Signal_Processing.md) / [PDF](Assets/Notes/Notes_PDF/2022_01_06-GuanyuHu-Audio_Signal_Processing.pdf) | [Code](Assets/Files/Codes/2022_01_06-GuanyuHu-Audio_Signal_Processing.7z) |   |
| 2022-02-24 | Week01 | Meta Learning                                    | 张与弛         | [Slide](Assets/Slides/2022_02_24-ZhangYuchi-meta_learning.pptx) / [PDF](Assets/Slides/Slides_PDF/2022_02_24-ZhangYuchi-meta_learning.pdf) | [Note](Assets/Notes/2022_02_24-ZhangYuchi-meta_learning.md) | -                                                                                |   |
| 2022-11-03 | Week09 | Diffusion Model                            | 宋怡馨    |- |[Note](Assets/Notes/2022_11_03-YixinSong-Diffusion_Model.md) / [PDF](Assets/Notes/Notes_PDF/2022_11_03-YixinSong-Diffusion_Model.pdf) | [Code](Assets/Files/Codes/2022_11_03-YixinSong-Diffusion_Model_demo.ipynb)| |
| 2022-11-10 | Week10 | SCAS1                            | 闫旭    |- |[Note](Assets/Notes/2022_11_10-YanXu-SCAS1.md) | [Vsdx](Assets/Files/Vsdx/2022_11_10-YanXu-SCAS1_FlowChart.vsdx)| |
| 2022-11-17 | Week11 | Text to Speech                                  | 侯玉亮       | [Slide ](Assets/Slides/2022_11_17-YuliangHou-Text_to_Speech.pptx) / [PDF](Assets/Slides/Slides_PDF/2022_11_17-YuliangHou-Text_to_Speech.pdf) | - | - ||
| 2022-11-17 | Week11 | Voice Conversion                            | 李徐梁    |- | [Note](Assets/Notes/Notes_PDF/2022_11_17-XuliangLi-Voice_Conversion.pdf) | -||
| 2021-12-01 | Week14 | Face Recognition and Uncertainty Learning                   | 曹至欣     | -                                                                                                                                                                       | [Note](Assets/Notes/2022_12_01-ZhixinCao-Face_Recognition_and_Uncertainty_Learning.md) | -                                                                                |   |
