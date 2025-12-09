# 规则

- 所有提交必须由本人独立完成。参赛队伍名称使用本人姓名，在[此页面](https://www.kaggle.com/competitions/face-super-resolution/team)设置。
- 提交的结果必须是由本人训练的**模型**生成的预测结果。
- **不可以**使用额外的训练数据集。
- **有限制地**可以使用预训练权重，可使用通用基础视觉模型（如：ViT）初始化参数。**！！！不可以完整地使用开源权重直接完成模型预测**。
- 违反上述规则，成绩将被判定为无效。
- Kaggle提交截至时间为**12月9日上午10:00**。
- 最后一周上课（12月16日）汇报你的方法，展示结果。
- 完整的训练、测试代码、预训练权重、推理结果需提交到BB平台，并撰写一份技术报告（中英文均可，字号11pt，双栏不少于4页，要有完整的方法介绍，正确引用参考文献），截止时间为**12月16日上午10:00**。

# 赛道1：弱监督的语义分割

## 请注意，前一版本的数据集分辨率有误，请重新在下面的链接下载数据集，重新测试（如果已经在先前的数据集上完成了训练，可以不用重新训练）！
- **北外网盘：[https://icloud.bfsu.edu.cn/f/110dac36950949daad1c/?dl=1](https://icloud.bfsu.edu.cn/f/110dac36950949daad1c/?dl=1)**

## 请注意 Convert.py 函数有更新，会显著影像kaggle得分计算！

**通过此链接进入竞赛主页：[https://www.kaggle.com/competitions/semantic-segmentation](https://www.kaggle.com/competitions/semantic-segmentation)**

  - [**Paraformer: Updating Large-scale High-resolution Geographical Maps from Limited Historical Labels**](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Learning_without_Exact_Guidance_Updating_Large-scale_High-resolution_Land_Cover_Maps_CVPR_2024_paper.pdf)
- [GitHub原始仓库](https://github.com/LiZhuoHong/Paraformer)

## 训练指南

1. 进入 Track1-WSSSeg 文件夹中
   ```bash
   cd ./Track1-WSSSeg
   ```

1. 下载预训练的ViT-B_16权重文件，放到 *"./networks/pre-train_model/imagenet21k"* 目录下。
   - [**谷歌Drive**](https://drive.google.com/file/d/10Ao75MEBlZYADkrXE4YLg6VObvR0b2Dr/view?usp=sharing)
   - [**北外网盘**](https://icloud.bfsu.edu.cn/f/dfd84e4d1c7b4feb9b4d/?dl=1)
   
2. 下载 Chesapeake_NewYork_dataset 数据集（约10.15GB），解压后保存到 *"./dataset/Chesapeake_NewYork_dataset"* 目录。
   - [**Kaggle Dataset（已更新，请使用最新版本）**](https://www.kaggle.com/datasets/pandawu/chesapeake-newyork-dataset)
   - [**北外网盘（已更新）**](https://icloud.bfsu.edu.cn/f/110dac36950949daad1c/?dl=1)

3. 配置环境，参考requirements.txt，建议直接在AutoDL上从预安装好pytorch2.0.0的镜像启动。
   
4. 运行训练脚本
   ```bash
   python train.py --dataset Chesapeake --batch_size 10 --max_epochs 100 --savepath experiments --gpu 0
   ```

5. 运行测试脚本
   ```bash
   python test.py --dataset Chesapeake --model_path *The path of trained .pth file* --save_path experiments/results --gpu 0
   ```

1. 将 .csv 文件下载到本地，提交到Kaggle平台。注意，Team的名称采用中文真实姓名。

# 赛道2：人脸图像恢复

人脸图像超分辨率是计算机视觉中的重要任务，旨在从低分辨率人脸图像中恢复高分辨率细节。本赛题聚焦于现实场景下的人脸复原，参赛者需要使用高效的超分辨率算法，将低质量的人脸图像增强为高质量的高分辨率图像。

**通过此链接进入竞赛主页：[https://www.kaggle.com/competitions/face-super-resolution](https://www.kaggle.com/competitions/face-super-resolution)**


## 准备训练数据集FFHQ512（约27.1G）
   - [**Kaggle Dataset**](https://www.kaggle.com/datasets/chelove4draste/ffhq-512x512)
   - [**北外网盘**](https://icloud.bfsu.edu.cn/f/94f37c1c8f99469a852c/?dl=1)

## 测试集和评测代码

1. 进入 [kaggle 竞赛主页](https://www.kaggle.com/competitions/face-super-resolution/data)
2. 测试集。 含100张经过一定程度降质后的人脸低分辨率图像（LR），用于测试模型的性能。
2. 评测代码在benchmark文件夹中，将 `compute_benchmark.py` 中的 `SUBMISSION_DIR` 修改为自己的结果路径，可以生成 `.csv` 文件。执行
   ```
   python compute_benchmark.py
   ```
1. 将 `.csv` 文件上传到kaggle平台，可以查看提交得分。提交前请检查文件中是否存在NaN值等错误。

## 训练和测试请参考Baseline

面向人脸恢复的模型非常多，提供一个Baseline作为快速上手的参考。[Restorerfomer++](https://github.com/wzhouxiff/RestoreFormerPlusPlus)
