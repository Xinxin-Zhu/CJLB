# CJLB
Code for NEUROCOMPUTING 2020 paper. "RGB-D Salient Object Detection via Cross-Modal Joint Feature Extraction and Low-Bound Fusion Loss".
# Overall
![avatar](https://github.com/Xinxin-Zhu/CJLB/tree/master/Figures/overall.png)
# Visual examples
![avatar](https://github.com/Xinxin-Zhu/CJLB/tree/master/Figures/visual_examples.png)
## CJLB Code

### > Requirment
+ caffe
+ PIL
+ numpy

### > Usage
#### 1. Clone the repo
```
git clone https://github.com/jiwei0921/DMRA.git
cd DMRA/
```
#### 2. Train/Test
+ test     
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), and put the test model [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), in the "**/models**". Meanwhile, you need to set relevant path correctly.
```
cd Test
python RGBD_test.py
```
+ train
The whole network training process includes two stages. In the first stage, a VGG16 model pre-trained [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), on ImageNet is used to initialize the parameters of RGB and depth saliency prediction streams respectively,
and the two independent streams are trained until convergence. 
```
cd Train
python run.py ../models/vgg16_RGB-Depth_pre_train.caffemodel RGBNet_train.prototxt(or DepthNet_train.prototxt)
```
In the second stage, the whole network is initialized by the weights of the two streams [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), , and the final model is obtained through further joint training.  
```
cd Train
python run.py ../models/RGBDNet_pre_train.caffemodel RGBDNet_train.prototxt
```   

### > Model evaluation

The following are the evaluation results of the model on five RGB-D datasets.

**Datasets\EvaluationMetrics**| F-measure | E-measure | S-measure | MAE(Low better) |    
:-: | :-: | :-: | :-: | :-: |  
NLPR | 0.887 | 0.949 | 0.906 | 0.033 |  
NJUD | 0.877 | 0.923 | 0.883 | 0.056 |
STEREO | 0.872 | 0.927 | 0.880 | 0.055 |  
LFSD | 0.807 | 0.856 | 0.832 | 0.106 |
DES | 0.898 | 0.962 | 0.910 | 0.030 |
 
+ Tips: **The results of the paper shall prevail.** Because of the randomness of the training process, the results fluctuated slightly.

### > Results  
| [NJUD](https://pan.baidu.com/s/15opVkn2QQ1DXttD2-h17AA)  |
| [NLPR](https://pan.baidu.com/s/1QHdWodsxknvXZb1YLDOUgA)  |
| [STEREO](https://pan.baidu.com/s/1UpUTEGS_1rayKwY5-LXKjw)  |
| [LFSD](https://pan.baidu.com/s/1G_x1g5ZaBTDNinS1IwOwgA)  |
| [DES](https://pan.baidu.com/s/1JJTWU9gObkvmEq0BVA57Qg)  |
+ Note:  The extraction code is XING.

  
### > Saliency maps of related models      
+ The web link is [here](http://dpfan.net/d3netbenchmark/).

### If you think this work is helpful, please cite
```
@InProceedings{Zhu_2020_NC,       
   author = {Yongri {Piao} and Wei {Ji} and Jingjing {Li} and Miao {Zhang} and Huchuan {Lu}},   
   title = {Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection},     
   booktitle = "ICCV",     
   year = {2020}     
}  
```
@article{Zhu_2020_NC, 
  title={RGB-D Salient Object Detection via Cross-Modal Joint Feature Extraction and Low-Bound Fusion Loss},
  author={Zhu, Xinxin and Li, Yi and Fu, Huazhu and Fan, Xiaoting and Shi, Yanan and Lei, Jianjun},
  journal={Neurocomputing},
  year={2020}
}
### Contact Us
If you have any questions, please contact us ( xxzhu@163.com or weiji.dlut@gmail.com ).
