# CJLB
Code for NEUROCOMPUTING 2020 paper. "RGB-D Salient Object Detection via Cross-Modal Joint Feature Extraction and Low-Bound Fusion Loss".
# Overall
![avatar](https://github.com/jiwei0921/DMRA/blob/master/figure/overall.png)


## CJLB Code

### > Requirment
+ pytorch 0.3.0+
+ torchvision
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
Download related dataset [**link**](https://github.com/jiwei0921/RGBD-SOD-datasets), and set the param '--phase' as "**test**" and '--param' as '**True**' in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.
```
python demo.py
```
+ train     
Our train-augment dataset [**link**](https://pan.baidu.com/s/18nVAiOkTKczB_ZpIzBHA0A) [ fetch code **haxl** ] / [train-ori dataset](https://pan.baidu.com/s/1B8PS4SXT7ISd-M6vAlrv_g), and set the param '--phase' as "**train**" and '--param' as '**True**'(loading checkpoint) or '**False**'(no loading checkpoint) in ```demo.py```. Meanwhile, you need to set **dataset path** and **checkpoint name** correctly.  
```
python demo.py
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

  
### > Related RGB-D Saliency Datasets
All common RGB-D Saliency Datasets we collected are shared in ready-to-use manner.       
+ The web link is [here](https://github.com/jiwei0921/RGBD-SOD-datasets).


### If you think this work is helpful, please cite
```
@InProceedings{Piao_2019_ICCV,       
   author = {Yongri {Piao} and Wei {Ji} and Jingjing {Li} and Miao {Zhang} and Huchuan {Lu}},   
   title = {Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection},     
   booktitle = "ICCV",     
   year = {2019}     
}  
```

### Contact Us
If you have any questions, please contact us ( jiwei521@mail.dlut.edu.cn or weiji.dlut@gmail.com ).
