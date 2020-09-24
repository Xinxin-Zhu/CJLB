#coding=utf-8
import caffe
import numpy as np
from PIL import Image
from scipy.misc import imresize
import matplotlib.pyplot as plt
import sys
import random
from numpy import *
import math
from scipy.misc import imresize
from scipy.misc import imrotate


class RGBDsal_dataaug(caffe.Layer):
    def setup(self, bottom, top):

        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.augment    = params.get('with_augmentation', True)
        self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 3:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx1 = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
        #数据增强

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]    #0.1 201.6-246.4
           self.max_rotate      = self.aug_params[2]
           #self.max_transW      = self.aug_params[3]
           #self.max_transH      = self.aug_params[4]
           self.flip            = (self.aug_params[3]>0)
        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        while True:
            # pick next input
            if self.random:
                self.idx1 = random.randint(0, len(self.indices)-1)
            else:
                self.idx1 += 1
                if self.idx1 == len(self.indices):
                    self.idx1 = 0

            idx1 = self.idx1

            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.data1= self.load_image1(self.indices[self.idx1].split(' ')[0])
               self.label1= self.load_label1(self.indices[self.idx1].split(' ')[1])
               #self.label1_1=np.expand_dims(self.label1,axis=0)
               self.depth= self.load_depth(self.indices[self.idx1].split(' ')[2])
              
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  (random.random()*2-1) * self.max_rotate
               #trans_w     =  np.int( (random.random()*2-1) * self.max_transW * self.W )
               #trans_h     =  np.int( (random.random()*2-1) * self.max_transH * self.H )
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.data1    = self.load_image_transform(self.indices[self.idx1].split(' ')[0],scale, rotation, flip)               
               self.label1   = self.load_label_transform(self.indices[self.idx1].split(' ')[1],scale, rotation, flip)
               #self.label1_1=np.expand_dims(self.label1,axis=0)
               self.depth    = self.load_depth_transform(self.indices[self.idx1].split(' ')[2],scale, rotation, flip)
            break

        #self.label2   = np.ones((1,224,224),dtype=np.uint8)
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.data1.shape)
        top[1].reshape(2, *self.label1.shape)
        #top[2].reshape(2, *self.label1_1.shape)
        top[2].reshape(2, *self.depth.shape)
        #top[4].reshape(2, *self.label2.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1
        #top[2].data[...] = self.label1_1
        top[2].data[...] = self.depth
        #top[4].data[...] = self.label2
        
    def backward(self, top, propagate_down, bottom):
        pass

    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        in_ = np.array(im, dtype=float32)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        in_ = in_.transpose((2,0,1))
        return in_   

    def load_depth(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #print('im.shape:',im.size)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #in_ = np.array(im, dtype=np.uint8)
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        #in_ = in_/255.0
        return in_             

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label


    def load_label2(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label 

    def load_image_transform(self, idx, scale, rotation, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) ) 

        #print >> sys.stderr, 'loading {}'.format(idx)
        #print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im    = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im    = im.resize((img_W,img_H))
        #im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H),Image.NEAREST)

        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean1  
        in_ = in_.transpose((2,0,1))
        return in_



    def load_label_transform(self, idx, scale, rotation, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) ) 

        #print >> sys.stderr, 'loading {}'.format(idx)
        #print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im    = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im    = im.resize((img_W,img_H))
        #im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H),Image.NEAREST)

        #plt.imshow(im,cmap='gray')
        #plt.show()

        label = np.array(im, dtype=np.uint8)
        label = label/255
        #print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        return label
 
    def load_depth_transform(self, idx, scale, rotation, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) ) 

        #print >> sys.stderr, 'loading {}'.format(idx)
        #print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im    = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im    = im.resize((img_W,img_H))
        #im    = im.transform((img_W,img_H),Image.AFFINE,(1,0,trans_w,0,1,trans_h))
        im    = im.rotate(rotation)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H),Image.NEAREST)

        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        return in_ 


class depthsal(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', 455)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 2:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()


        # make eval deterministic
        if 'Img_depth_train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.depth= self.load_image1(self.indices[self.idx].split(' ')[2])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.depth.shape)
        top[1].reshape(2, *self.label1.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.depth
        top[1].data[...] = self.label1


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #print('im.shape:',im.size)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #in_ = np.array(im, dtype=np.uint8)
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        #in_ = in_/255.0
        return in_        

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        label = np.array(im, dtype=np.uint8)
        label = label/255.0
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label  

class depthsal_val(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 2:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_val' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.depth= self.load_image1(self.indices[self.idx].split(' ')[2])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.depth.shape)
        top[1].reshape(2, *self.label1.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.depth
        top[1].data[...] = self.label1


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #print('im.shape:',im.size)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #in_ = np.array(im, dtype=np.uint8)
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        #in_ = in_/255.0
        return in_        

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255.0
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label


class RGBDsal(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 3:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx1 = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.data1= self.load_image1(self.indices[self.idx].split(' ')[0])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1])
        self.depth= self.load_depth(self.indices[self.idx].split(' ')[2])
        #self.label2= self.load_label2(self.indices[self.idx].split(' ')[3])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.data1.shape)
        top[1].reshape(2, *self.label1.shape)
        top[2].reshape(2, *self.depth.shape)
        #top[3].reshape(2, *self.label2.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1
        top[2].data[...] = self.depth
        #top[3].data[...] = self.label2


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        in_ = np.array(im, dtype=float32)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        in_ = in_.transpose((2,0,1))
        return in_   

    def load_depth(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #print('im.shape:',im.size)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #in_ = np.array(im, dtype=np.uint8)
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        #in_ = in_/255.0
        return in_      

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        label = label[np.newaxis, ...]
        return label


    def load_label2(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label  

class RGBDsal_val(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 3:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx1 = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_val' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.data1= self.load_image1(self.indices[self.idx].split(' ')[0])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1])
        self.depth= self.load_depth(self.indices[self.idx].split(' ')[2])
        #self.label2= self.load_label2(self.indices[self.idx].split(' ')[3])

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data1.shape)
        top[1].reshape(1, *self.label1.shape)
        top[2].reshape(1, *self.depth.shape)
        #top[3].reshape(1, *self.label2.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1
        top[2].data[...] = self.depth
        #top[3].data[...] = self.label2


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        in_ = np.array(im, dtype=float32)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        in_ = in_.transpose((2,0,1))
        return in_   

    def load_depth(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #print('im.shape:',im.size)
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        #in_ = np.array(im, dtype=np.uint8)
        in_ = np.expand_dims(in_,axis=0)
        in_ -= self.mean2
        #in_ = in_/255.0
        return in_             

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label 

    def load_label2(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label
class RGBsal(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 2:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.data1= self.load_image1(self.indices[self.idx].split(' ')[0])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1]) 
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.data1.shape)
        top[1].reshape(2, *self.label1.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=float32)
        #print (in_)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        #print (in_)
        in_ = in_.transpose((2,0,1))
        return in_        

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #plt.imshow(im,cmap='gray')
        #plt.show()
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #print(label)
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label      

class RGBsal_val(caffe.Layer):

    def setup(self, bottom, top):

        # config
        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        #self.augment    = params.get('with_augmentation', True)
        #self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 2:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()


        # make eval deterministic
        if 'Img_depth_val' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):

        self.data1= self.load_image1(self.indices[self.idx].split(' ')[0])
        self.label1= self.load_label1(self.indices[self.idx].split(' ')[1]) 
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data1.shape)
        top[1].reshape(1, *self.label1.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1


        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        in_ = np.array(im, dtype=float32)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        in_ = in_.transpose((2,0,1))
        return in_        

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label

class RGBsal_dataaug(caffe.Layer):

    def setup(self, bottom, top):

        params = eval(self.param_str)
        self.RGBD_data_dir = params['RGBD_data_dir']
        self.split     = params['split']
        self.mean1      = np.array(params['mean1'])
        self.mean2      = np.array(params['mean2'])
        self.random    = params.get('randomize', True)
        self.seed      = params.get('seed', None)
        self.scale     = params.get('scale', 1)
        self.augment    = params.get('with_augmentation', True)
        self.aug_params = np.array(params['aug_params']) #( aug_num, max_scale, max_rotate, max_translation, flip)
        self.H         = 224
        self.W         = 224

        if len(top) != 2:
            raise Exception("Need to define three tops: data1,data2, label1")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/{}.txt'.format(self.RGBD_data_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()

        self.idx1 = 0 # we pick idx in reshape

        # make eval deterministic
        if 'Img_depth_train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)
        #数据增强

        if self.augment:
           self.aug_num         = np.int(self.aug_params[0])
           self.max_scale       = self.aug_params[1]    #0.1 201.6-246.4
           #self.max_rotate      = self.aug_params[2]
           self.flip            = (self.aug_params[2]>0)
        # make eval deterministic
        if 'train' not in self.split:
            self.random = False

        # randomization: seed and pick
        if self.random:
            random.seed(self.seed)
            self.idx = random.randint(0, len(self.indices)-1)


    def reshape(self, bottom, top):
        while True:
            # pick next input
            if self.random:
                self.idx1 = random.randint(0, len(self.indices)-1)
            else:
                self.idx1 += 1
                if self.idx1 == len(self.indices):
                    self.idx1 = 0

            idx1 = self.idx1

            if self.augment == False or random.randint(0, self.aug_num) == 0:
               self.data1= self.load_image1(self.indices[self.idx1].split(' ')[0])
               self.label1= self.load_label1(self.indices[self.idx1].split(' ')[1])
            else:
               scale       =  (random.random()*2-1) * self.max_scale
               rotation    =  random.randint(0,180)
               if self.flip:
                  flip     = (random.randint(0,1) > 0)
               else:
                  flip     = False
               self.data1    = self.load_image_transform(self.indices[self.idx1].split(' ')[0],scale, rotation,flip)
               self.label1   = self.load_label_transform(self.indices[self.idx1].split(' ')[1],scale, rotation,flip)
            break
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(2, *self.data1.shape)
        top[1].reshape(2, *self.label1.shape)
        
    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data1
        top[1].data[...] = self.label1
        
        
    def backward(self, top, propagate_down, bottom):
        pass


    def load_image1(self, idx):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=float32)
        #print (in_)
        in_ = in_[:,:,::-1]
        #print(in_.size())
        in_ -= self.mean1
        #print (in_)
        in_ = in_.transpose((2,0,1))
        return in_        

    def load_label1(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        #print >> sys.stderr, 'loading {}'.format(idx)
        im = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im  = im.resize((self.W, self.H))
        #plt.imshow(im,cmap='gray')
        #plt.show()
        #label = np.array(im, dtype=np.float32)
        label = np.array(im, dtype=np.uint8)
        label = label/255
        #print(label)
        #label = np.uint8((label>0))
        #label = label[np.newaxis, ...]
        return label      

    def load_image_transform(self, idx, scale, rotation, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) ) 

        #print >> sys.stderr, 'loading {}'.format(idx)
        #print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im    = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im    = im.resize((img_W,img_H))
        im    = im.rotate(rotation)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H),Image.NEAREST)

        #plt.imshow(im,cmap='gray')
        #plt.show()
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:,:,::-1]
        in_ -= self.mean1  
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label_transform(self, idx, scale, rotation, flip):
        img_W = np.int( self.W*(1.0 + scale) )
        img_H = np.int( self.H*(1.0 + scale) ) 

        #print >> sys.stderr, 'loading {}'.format(idx)
        #print >> sys.stderr, 'scale: {}; rotation: {}; translation: ({},{}); flip: {}.'.format(scale, rotation, trans_w, trans_h, flip)

        im    = Image.open('{}/{}'.format(self.RGBD_data_dir, idx))
        im    = im.resize((img_W,img_H))
        im    = im.rotate(rotation)
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
       
        if scale>0:
           box = (np.int((img_W - self.W)/2), np.int((img_H - self.H)/2), np.int((img_W - self.W)/2)+self.W, np.int((img_H - self.H)/2)+self.H)
           im  = im.crop(box)
        else:
           im  = im.resize((self.W, self.H),Image.NEAREST)

        #plt.imshow(im,cmap='gray')
        #plt.show()

        label = np.array(im, dtype=np.uint8)
        label = label/255
        #print >> sys.stderr, 'Number of Objects: {}'.format(np.max(label))
        
        return label

