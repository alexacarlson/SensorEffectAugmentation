from __future__ import division
import matplotlib
matplotlib.use('Agg')
#
import datetime
import os
import shutil
from glob import glob
#
import pdb
import math
import numpy as np
import time
import tensorflow as tf
from six.moves import xrange
#
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as sio
import skimage
import skimage.transform
import skimage.data
import skimage.filters
import scipy
import scipy.misc
import scipy.stats
import scipy.signal
from scipy.spatial.distance import cdist
#
from augmentfunctions_tf import *

################################################################################
############################## camGAN class ####################################  
################################################################################  


class camGAN(object):
  def __init__(self, sess, image_height, image_width, batch_size, channels, 
              input, colour_shift, chromatic_aberration, blur, exposure, noise, save_params,
              pattern, output):
    """
    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
    """
    self.sess = sess
    self.batch_size = batch_size
    self.output = output
    ## Dataset info
    self.G_dataset = input
    self.G_output_height = image_height
    self.G_output_width = image_width
    self.channels = channels
    self.pattern = pattern
    ## Augmentation Flags
    self.save_params = save_params
    self.chromatic_aberration = chromatic_aberration
    self.blur = blur
    self.exposure = exposure
    self.noise = noise
    self.colour_shift = colour_shift
    ##
    ## Build the model/graph
    self.build_model()

  def build_model(self):
    ##
    ## construct the graph of the image augmentation architecture
    ##
    print('building model/graph')
    ## initialize graph input palceholders
    image_dims = [self.G_output_height, self.G_output_width, self.channels]
    self.G_inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='G_input_images')
    G_inputs = self.G_inputs
    #
    ## Camera generator graph ##
    self.aug_image_genOP = self.generate_augmentation(G_inputs)
    #

  def augment_batches(self, config):
    ##
    ## augments the dataset in batches. Can augment the dataset multiple times by specifying n >1 (i.e., n = 1 augments each image once)
    ##
    save_params = self.save_params
    # get file list of data/labels to augment, get batches
    G_data = sorted([fn for fn in glob(os.path.join(config.input, config.pattern)) if 'aug' not in fn])
    N = len(G_data)
    batch_idxs = N // config.batch_size
    randombatch = np.arange(batch_idxs*config.batch_size)
    print("Size of dataset to be augmented: %d"%(len(G_data)))
    #
    begin_n=0
    for n in xrange(begin_n, config.n):
      #
      for idx in xrange(0, (batch_idxs*config.batch_size), config.batch_size):
        ##
        ## generate a batch of num_augs for the image
        G_batch_images, G_batch_files = self.load_data_batches(G_data, config.batch_size, randombatch, idx)
        #
        ## Augment data by sampling from a random dist. and pushing images through the generator
        out = self.sess.run([self.aug_image_genOP], feed_dict={self.G_inputs: G_batch_images})
        #
        ## generator output images and sampled augmentation parameters
        G_output_images = np.squeeze(out[0][0])
        ChromAbParams = np.array(out[0][1])
        BlurParams =   np.array(out[0][2])
        ExpParams =   np.array(out[0][3])
        NoiseParams = np.array(out[0][4])
        ColorParams = np.array(out[0][5])
        ## save images
        self.save_augmented_final_images(G_output_images, G_batch_files, ChromAbParams, BlurParams, ExpParams, NoiseParams, ColorParams, n)

  ## ---------------------------------------------------------------- ##
  ## ---- IMAGE AUGMENTATION PIPELINE (and supporting functions) ---- ##
  ## ---------------------------------------------------------------- ##
  def generate_augmentation(self, imageBatch):
    ##
    ## Augments an image batch using physically based model of camera effects during image formation process.
    ## Augmentation parameters are uniformly sampled from specified ranges that yeild visually realistic results.
    ##
    crop_h = self.G_output_height
    crop_w = self.G_output_width
    batchsize = self.batch_size 
    AugImg = imageBatch
    #
    save_params = self.save_params
    chromatic_aberration = self.chromatic_aberration
    blur = self.blur
    exposure = self.exposure
    noise = self.noise
    colour_shift = self.colour_shift
    #
    # Chromatic Aberration ##
    if chromatic_aberration:
      # augment with chromatic aberration
      scale_val = tf.random_uniform((batchsize,1,1,1), minval = 0.998, maxval = 1.002, dtype=tf.float32)
      minT = -0.002
      maxT = 0.002
      tx_Rval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)
      ty_Rval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)
      tx_Gval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)  
      ty_Gval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)
      tx_Bval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)  
      ty_Bval = tf.random_uniform((batchsize,1,1,1), minval=minT, maxval = maxT, dtype=tf.float32)
      AugImg = aug_chromab(AugImg, crop_h, crop_w, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval)
    else:
      scale_val = []
      tx_Rval = []
      ty_Rval = []
      tx_Gval = [] 
      ty_Gval = []
      tx_Bval = [] 
      ty_Bval = []

    ## Blur ##
    if blur:
      #augment the image with blur
      window_h = tf.random_uniform((batchsize,1), minval=3.0, maxval=11.0,dtype=tf.float32)
      sigmas = tf.random_uniform((batchsize,1), minval=0.0, maxval=3.0,dtype=tf.float32) # uniform from 0 to 1.5
      AugImg = aug_blur(AugImg, window_h, sigmas, batchsize)
    else:
      window_h = []
      sigmas = []

    ## Exposure ##
    if exposure:
      # augment image with exposure
      delta_S = tf.random_uniform((batchsize,1,1,1), minval=-0.6, maxval=1.2, dtype=tf.float32)
      A = 0.85
      A_S = tf.constant(A,shape=(batchsize,1,1,1),dtype=tf.float32)
      AugImg = aug_exposure(AugImg, delta_S, A_S, batchsize)
    else:
      delta_S = []

    ## Sensor Noise ## 
    if noise:
      # augment image with sensor noise
      N=0.001
      Ra_sd = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      Rb_si = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      Ga_sd = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      Gb_si = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      Ba_sd = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      Bb_si = tf.random_uniform((batchsize,1,1,1), minval=0.0, maxval=N, dtype=tf.float32)
      AugImg = aug_noise(AugImg,batchsize,Ra_sd, Rb_si, Ga_sd,Gb_si, Ba_sd, Bb_si, crop_h, crop_w)
    else:
      Ra_sd = []
      Rb_si = []
      Ga_sd = []
      Gb_si = []
      Ba_sd = []
      Bb_si = []

    ## Color shift/Tone mapping ##
    if colour_shift:
      # augment image by shifting color temperature
      a_transl = tf.random_uniform((batchsize,1,1,1),minval=-30.0, maxval=30.0,dtype=tf.float32)
      b_transl = tf.random_uniform((batchsize,1,1,1),minval=-30.0, maxval=30.0,dtype=tf.float32)
      AugImg = aug_color(AugImg, a_transl, b_transl)
    else:
      a_transl = []
      b_transl = []

    if save_params:
      ## Log the sampled augmentation parameters
      ChromAbParams = [tf.squeeze(scale_val), tf.squeeze(tx_Rval), tf.squeeze(ty_Rval), tf.squeeze(tx_Gval), tf.squeeze(ty_Gval), tf.squeeze(tx_Bval), tf.squeeze(ty_Bval)]
      BlurParams = [tf.squeeze(window_h), tf.squeeze(sigmas)]
      ExpParams = tf.squeeze(delta_S)
      NoiseParams = [tf.squeeze(Ra_sd), tf.squeeze(Rb_si), tf.squeeze(Ga_sd), tf.squeeze(Gb_si), tf.squeeze(Ba_sd), tf.squeeze(Bb_si)]
      ColorParams = [tf.squeeze(a_transl), tf.squeeze(b_transl)]
      return AugImg, ChromAbParams, BlurParams, ExpParams, NoiseParams, ColorParams 
    else:
      return AugImg

  ## ---------------------------- ##
  ## ---- utility functions) ---- ##
  ## ---------------------------- ##
  def read_img(self, filename):
    imgtmp = scipy.misc.imread(filename)
    ds = imgtmp.shape
    ## remove any depth channel
    if ds[2]>self.channels:
      imgtmp = np.squeeze(imgtmp[:,:,:self.channels])
    ## resize image to specified height and width
    img = scipy.misc.imresize(imgtmp,(self.G_output_height,self.G_output_width,3))
    img = np.array(img).astype(np.float32)
    return img

  def load_data_batches(self, data, batch_size, randombatch, idx):
    ##
    ## loads in images and resizes to all the same size
    ##
    batch_files = []
    batch_labels=[]
    for id in xrange(0, batch_size):
        batch_files = np.append(batch_files, data[randombatch[idx+id]])
    ## center cropping
    #batch=[]
    #for batch_file in batch_files:
    #  Im = scipy.misc.imread(batch_file)
    #  y,x,c = Im.shape
    #  cropx = 1914
    #  cropy = 1046
    #  startx = x//2-(cropx//2)
    #  starty = y//2-(cropy//2)    
    #  if y > 1046:
    #    Im = Im[starty:starty+cropy,startx:startx+cropx]   
    #  if y < 1046:
    #    self.bad_image.append(batch_file) 
    #    Im = scipy.misc.imresize(Im,(cropy,cropx),'cubic')
    #  batch.append(Im)
    #s
    batch_images = [self.read_img(batch_file) for batch_file in batch_files]
    #
    return batch_images, batch_files

  def save_augmented_final_images(self, output_images, batch_files, ChromAbParams, BlurParams, ExpParams, NoiseParams, ColorParams, n):
    print(self.save_params,self.chromatic_aberration,self.blur,self.exposure,self.noise,self.colour_shift)
    ##
    save_params = self.save_params
    ##
    for img_idx in range(0,self.batch_size):
        # get image
        image_out = output_images[img_idx]
        image_out_file = batch_files[img_idx]
        # generate fileID and paths
        imID = os.path.splitext(os.path.split(image_out_file)[1])[0]
        out_name = os.path.join(self.output, imID+'_aug_'+str(n+1)+'.png')
        #out_name = os.path.join(self.output, imID+'_aug.png')
        try:
            ## save the image
            image_save = np.squeeze(image_out)
            ## clip and save the augmented image
            image_save[image_save > 255.0] = 255.0
            image_save[image_save < 0.0] = 0.0
            image_save = Image.fromarray((image_save).astype(np.uint8))
            print("saved %s to results directory"%(out_name))
            image_save.save(out_name)
            ##
            if save_params:
              ## save the augmentation parameters for the image
              if ChromAbParams.any():
                chromabP = 'chromab,'+','.join([str(x) for x in ChromAbParams[:,img_idx]])
              else:
                chromabP=''
              if BlurParams.any():
                blurP = 'blur,'   + ','.join([str(x) for x in BlurParams[:,img_idx]])
              else:
                blurP = ''
              if ExpParams.any():
                expP = 'exposure,' + str(ExpParams[img_idx])
              else:
                expP=''
              if NoiseParams.any():
                noiseP = 'noise,' + ','.join([str(x) for x in NoiseParams[:,img_idx]])
              else:
                noiseP = ''
              if ColorParams.any():
                colorP = 'color,' + ','.join([str(x) for x in ColorParams[:,img_idx]])
              else:
                colorP = ''
              param_str='\n'.join([chromabP, blurP, expP, noiseP, colorP])
              fobj = open(os.path.splitext(out_name)[0]+'.txt','w')
              fobj.write(param_str)
              fobj.close()
        except OSError:
            print(out_name)
            print("ERROR!")
            pass
            #
## EOF ##