# -*- coding: utf-8 -*-
import os,sys
sys.path.append("/home/lab549/zxx/RGBD_Sal/caffe")
sys.path.append("/home/lab549/zxx/RGBD_Sal/caffe/python")
sys.path.append("/home/lab549/zxx/RGBD_Sal/caffe/python/caffe")
sys.path.insert(0,"../fcn_python/")

import caffe
import surgery
import numpy as np
from PIL import Image
import scipy.io
from scipy.misc import imresize
from scipy import io
import shutil

def load_image1(im_name_1):
      # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
    im = Image.open(im_name_1)
    im = im.resize((224,224))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    #in_ -= np.array((101.165369898,108.772791374,113.836235651))
    in_ -= np.array((104.508043305,110.817120576,114.231514496))
    in_ = in_.transpose((2,0,1))
    print >> sys.stderr, 'loading {}'.format(im_name_1)
    return in_

def load_depth(depth_name):

    im = Image.open(depth_name)
    im_  = im.resize((224, 224))
    in_ = np.array(im_, dtype=np.float32)
    in_ = np.expand_dims(in_,axis=0)
    #in_ -= np.array((130.20743649))
    in_ -= np.array((127.642412568))
    #in_ = in_[:,:,::-1] 
    #in_ = in_.transpose((2,0,1))deploy_RGBD_sal_VGG16_three_stream_res2net
    print >> sys.stderr, 'loading {}'.format(depth_name)
    return in_

deploy_proto = '../models/RGBD_test.prototxt'
device_id    = 0
caffe.set_device(device_id)
caffe.set_mode_gpu()
models_list=[]
caffe_model_path  = '../models/RGBD_train_models'
false_models=  os.listdir(caffe_model_path)
false_models.sort()

img_path1=  '../data/Train_data/Img/Img_test'
depth_path= '../data/Train_data/depth/depth_test'
images1    = os.listdir(img_path1)
images1.sort()
depth_dir  = os.listdir(depth_path)
depth_dir.sort()
for tt in false_models:
    if tt.endswith(".caffemodel"):
        write_name = tt
        models_list.append(write_name)

for idx2 in range(len(models_list)):
    caffe_model = '{}/{}'.format(caffe_model_path,models_list[idx2])
    fileout_name= models_list[idx2].split('_',5)[5]
    fileout_name= 'iter_' + fileout_name.split('.',1)[0]
    fileout_name= '../results/RGBD_results/'+ fileout_name
    os.mkdir(fileout_name)
    nju='nju'
    rgbd='rgbd'
    fileout1_name= fileout_name +'/nju'
    fileout2_name= fileout_name + '/rgbd'
    os.mkdir(fileout1_name)
    os.mkdir(fileout2_name)
    file_out1     = fileout1_name
    file_out2     = fileout2_name
    net = caffe.Net(deploy_proto , caffe_model, caffe.TEST)

    for idx in range(785):

        im_name_1 = '{}/{}'.format(img_path1,images1[idx])
        depth_name= '{}/{}'.format(depth_path,depth_dir[idx])
        ss = images1[idx].split('.jpg')
        ss = ss[0]
        if "left" in ss:
            seg_name    = '{}/{}.png'.format(file_out1, ss)
        else:
            seg_name    = '{}/{}.png'.format(file_out2, ss)
        data1 = load_image1(im_name_1)
        depth = load_depth(depth_name)

        net.blobs['data1'].reshape(1,  *data1.shape) 
        net.blobs['data1'].data[...]  = data1
        net.blobs['depth'].reshape(1,  *depth.shape)
        net.blobs['depth'].data[...]  = depth
 
        net.forward()
        out1 = net.blobs['C'].data[0][0,:,:]
        out1 = np.array(out1 * 255, dtype=np.uint8)
        res_img = Image.fromarray(out1)
        res_img.convert('L').save(seg_name)

print('done')