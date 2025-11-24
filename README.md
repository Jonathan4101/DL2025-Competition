# 赛道1：弱监督的语义分割

**本题目来自下面的工作，参考代码也来自作者的官方Baseline，感谢！**
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
   
2. 下载 Chesapeake_NewYork_dataset 数据集（约19.6GB），解压后保存到 *"./dataset/Chesapeake_NewYork_dataset"* 目录。
   - [**Kaggle Dataset**](https://www.kaggle.com/datasets/pandawu/chesapeake-newyork-dataset)
   - [**北外网盘**](https://icloud.bfsu.edu.cn/f/f951f210122643c8816d/?dl=1)

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

