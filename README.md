中文

## MLDA-Net深度估计模型

## 目录
[TOC]


## 1、简介

该模型的目标生成具有更完整边界和丰富深度细节的深度图。详情可以参考论文[原文](https://ieeexplore.ieee.org/abstract/document/9416235)。



![img](https://ai-studio-static-online.cdn.bcebos.com/89e3395703fa493a812f268768661a8bd7a4d50886dc4b018d9f140b454380d2)

该模型的总体结构如上图所示。最左边一列为多个不同尺度的输入数据，该尺度为可选参数scales。输入数据过两路卷积网络来提取特征，接着用注意网络GA进行整合并进一步提取特征。这部分就是该模型的第一个网络结构，在代码中名为models["encoder"]。

在提取特征后，该模型就进入第二个网络结构models["depth"]，即为上图右侧一半结构。该网络主要基于两种注意力块进行特征提取和上采样，最终输出不同尺度输入图所对应的不同尺度的深度图。



**模型输出示例：**

![img](https://ai-studio-static-online.cdn.bcebos.com/5698cba1880b444ea45573d741b87e68a80f471224324b568d43875502174ef8)





本项目也集成到了百度飞浆AI Studio中，可更快进行复现。

地址：[【第六届论文复现赛103题】 MLDA-Net深度估计模型 paddle复现 - 飞桨AI Studio (baidu.com)](https://aistudio.baidu.com/aistudio/projectdetail/4144881?contributionType=1)




## 2、准备工作

### 2.1 数据集准备

该模型的训练需要[kitti数据集](https://github.com/nianticlabs/monodepth2)，并且需要生成[depth_hint](https://github.com/nianticlabs/depth-hints)。可以参考链接来获取完整的数据集，并生成相应的depth_hint。完整的数据集有176GB的大小，本项目也提供了较小规模的kitti数据集，该数据集是kitti数据集中的10_03部分，并提供了生成的depth_hint进行简化训练。



链接：https://pan.baidu.com/s/1Rj5biYkAR2pURuzR881GQw ， 提取码：n5wa 



**下载后，文件组织形式如下**

```
|-- data/kitti_data
	|-- 2011_10_03
		|-- 2011_10_03_drive_0027_sync
			|-- image_00
			|-- image_01
			|-- image_02
			|-- ...
		|-- 2011_10_03_drive_0034_sync
			|-- ......
		|-- 2011_10_03_drive_0042_sync
			|-- ......
		|-- 2011_10_03_drive_0047_sync
		    |-- ......
		|-- calib_cam_to_cam.txt
		|-- calib_imu_to_velo.txt
		|-- calib_velo_to_cam.txt
```



### 2.2 模型准备

**模型参数文件下载地址：**

链接：https://pan.baidu.com/s/19Nl8d_SUOO-ihhgxegtE6w    提取码：owf6 

从链接中下载模型参数,并放到项目根目录下的data文件夹下，这样data文件夹的文件结构如下：

**文件结构**


```
  data/
        |-- lite_data              #predict和infer用的测试图片
        |-- lite_train_data        #少量训练数据，用于tipc进行小批量训练和测试
        |-- kitti_data             #完整训练数据
        |-- pretrain_weights       #预训练的的模型参数文件
            |-- depth.pdparams
            |-- encoder.pdparams
            |-- pose.pdparams
            |-- pose_encoder.pdparams
```



**复现精度(192 x 640 分辨率)：**

| Backbone | Train dataset     | Test dataset    | RMSE        | checkpoints_dir  |
| -------- | ----------------- | --------------- | ----------- | ---------------- |
| MLDA     | kitti/10_03/train | kitti/10_03/val | 4.690       |                  |
| MLDA     | kitti/10_03/train | kitti/10_03/val | 4.216(复现) | pretrain_weights |



## 3、开始使用

### 3.1 模型训练

对模型进行训练时，在控制台输入代码：

 ```shell
 python train.py --data_path ../data --depth_hint_path ../data/depth_hints
 ```

训练过程中会在MLDA-Net-repo/log_train/文件夹下生成train.log文件夹，用于保存训练日志。 

模型训练需使用paddle2.2版本，paddle2.3版本由于paddle.cumsum函数存在问题，会输出错误结果。




### 3.2 模型评估

对模型进行评估时，在控制台输入以下代码：

 ```shell
python test.py --data_path ./data/kitti_data --depth_hint_path ./data/kitti_data/depth_hints --load_weights_folder ./data/pretrain_weights
 ```

如果要在自己提供的模型上进行测试，请将修改参数 --load_weights_folder your_weight_folder



### 3.3 模型单图像测试

对模型进行单图像的简单测试时，在控制台输入以下代码。

 ```shell
python predict.py --load_weights_folder ./data/pretrain_weights

#如果要在自己提供的模型上进行测试，load_weights_folder。 如果要在自己提供的图片上进行测试，在控制台按如下格式输入代码。
python predict.py --load_weights_folder ./data/pretrain_weights  --color_path your_img_path --no_rmse True
 ```

输出示例如下:

```
predict_img saved to ./predict_figs/depth_predict.jpg
eval
   abs_rel |   sq_rel |     rmse | rmse_log |       a1 |       a2 |       a3 | 
&   0.120  &   0.765  &   4.355  &   0.170  &   0.848  &   0.973  &   0.992  \\
-> Done!
```

模型输出的图片与第一部分的图2相同。



## 4. Tipc

### 4.1 导出inference模型

```bash
python export.py --load_weights_folder ./data/pretrain_weights --save_dir ./inference
```

上述命令将生成预测所需的模型结构文件`model.pdmodel`和模型权重文件`model.pdiparams`以及`model.pdiparams.info`文件，均存放在`inference/`目录下。

由于该模型的主体实际上是两组模型，所以会生成两组文件model_encoder和model_depth。



### 4.2 使用预测引擎推理

```bash
python inference.py
```

推理结束会默认保存下模型生成的修复图像，并输出测试得到的RMSE值。效果与3.3相似。



### 4.3 调用脚本完成训推一体测试

#### 4.3.1 数据准备

从以下链接下载少量训练数据用于tipc训练，文件组织形式参考2.2。

链接：https://pan.baidu.com/s/1_XelODK9nDVc8YLKCADC1w  提取码：73mw 


#### 4.3.2 开始测试

测试基本训练预测功能的`lite_train_lite_infer`模式，运行：

```shell
# 运行测试
bash test_tipc/test_train_inference_python.sh ./test_tipc/configs/GPEN/train_infer_python.txt 'lite_train_lite_infer'
```



## 5. 代码结构说明


```
MLDA-Net-repo/ 
    |-- data/              #存放一些测试数据的文件夹
    |-- dataset/            #存放数据预处理相关的代码
    |-- networks/            #存放模型结构相关的代码
    |-- losses/             #存放损失函数计算相关的代码
    |-- utils/              #存放一些工具函数和类
    |-- test_tipc/           #存放tipc相关文件
    |-- splits/             #存放训练和测试数据列表
    |-- log_train/           #训练时生成的文件夹，用于存放训练过程中保存的模型参数和测试图片
      |-- trainer.py/         #模型训练和测试用到的类
      |-- train.py/          #模型训练时调用
      |-- test.py           #模型评估时调用
      |-- predict.py         #用模型测试单张图片时调用
      |-- export.py          #tipc生成推理模型时调用
      |-- inference.py        #tipc进行推理时调用
      |-- readme.md          #项目说明文档
```



## 6、参考文献与链接

论文地址：https://ieeexplore.ieee.org/abstract/document/9416235

论文复现指南-CV方向：https://github.com/PaddlePaddle/models/blob/release%2F2.2/tutorials/article-implementation/ArticleReproduction_CV.md

readme文档模板：https://github.com/PaddlePaddle/models/blob/release/2.2/community/repo_template/README.md