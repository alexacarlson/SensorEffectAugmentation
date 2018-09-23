##
## code for debugging augment layers
##
#from __future__ import division
import tensorflow as tf
import numpy as np
import os
import random
import math
import time
from geometric_transformation_module import perform_aff_transformation
from pix2pix_labtoRGBconv import *
import pdb
#
#     
# ---------------------------------------------------------------- #
# lens distortion augmentation functions
# ---------------------------------------------------------------- #
def aug_chromab(image, crop_h, crop_w, scale_val, tx_Rval, ty_Rval, tx_Gval, ty_Gval, tx_Bval, ty_Bval):
    #def aug_chromab(image, wlRarr, wlGarr, wlBarr, sigRarr, sigGarr, sigBarr, batchsize):
    #
    # adjust either longitudinal or lateral chromatic aberration
    # "longtudinal": shift in the direction of the optical axis (scale channels or blur)
    #     - set the value of either scale_x or scale_y set on the order of 0.005
    # "lateral": shift perpendicular to the optical axis, in the plane of the sensor or film (shift color channels) 
    #     - set the value of either tx or ty set on the order of 0.005
    #
    # longitudinal chromatic aberration: scale the green channel
    # lateral chromatic aberration: translate the channels
    #

    # normalize image to 0-1 range, convert to float
    image_ = tf.image.convert_image_dtype(image/255.0, tf.float32)
    # split the image into its channels
    R,G,B = tf.split(image_, 3, axis=3)
    #
    ## METHOD: IMAGE CHANNEL WARPING
    ## the below code generates a specific Hi for each channel
    ## red channel parameters
    R_alpha1 = tf.ones_like(tx_Rval)     #scale_r*math.cos(theta_r) # sx*r1
    R_alpha2 = tf.zeros_like(tx_Rval)     #-shear_r*math.sin(theta_r)# Sx*r2
    R_alpha3 = tx_Rval #tx
    R_alpha4 = tf.zeros_like(tx_Rval)    #shear_r*math.sin(theta_r) #sy*r4
    R_alpha5 = tf.ones_like(tx_Rval)    #scale_r*math.cos(theta_r) # Sy*r3  
    R_alpha6 = ty_Rval #ty
    # green channel parameters
    G_alpha1 = scale_val  #scale_g*math.cos(theta_g) # sx*r1
    G_alpha2 = tf.zeros_like(tx_Gval)        #-shear_g*math.sin(theta_g)# Sx*r2
    G_alpha3 = tx_Gval    #tx 
    G_alpha4 = tf.zeros_like(tx_Rval)        #shear_g*math.sin(theta_g) # sy*r4
    G_alpha5 = scale_val  #scale_g*math.cos(theta_g) # Sy*r3
    G_alpha6 = ty_Gval    #ty
    # blue channel parameters
    B_alpha1 = tf.ones_like(tx_Rval)     #scale_b*math.cos(theta_b) # sx*r1
    B_alpha2 = tf.zeros_like(tx_Bval)    #-shear_b*math.sin(theta_b)# Sx*r2
    B_alpha3 = tx_Bval                   #tx  
    B_alpha4 = tf.zeros_like(tx_Bval)    #shear_b*math.sin(theta_b) # sy*r4
    B_alpha5 = tf.ones_like(tx_Rval)     #scale_b*math.cos(theta_b) # Sy*r3 
    B_alpha6 = tx_Bval # ty
    ##
    num_aff_params = 6
    HRt = tf.stack([R_alpha1, R_alpha2, R_alpha3, R_alpha4, R_alpha5, R_alpha6], axis = 1)
    HGt = tf.stack([G_alpha1, G_alpha2, G_alpha3, G_alpha4, G_alpha5, G_alpha6], axis = 1)
    HBt = tf.stack([B_alpha1, B_alpha2, B_alpha3, B_alpha4, B_alpha5, B_alpha6], axis = 1)
    #
    # use the  spatial transformer layer to perform the affine warping on each channel for each image in the image batch
    augR = perform_aff_transformation(R, HRt, (crop_h, crop_w))
    augG = perform_aff_transformation(G, HGt, (crop_h, crop_w))
    augB = perform_aff_transformation(B, HBt, (crop_h, crop_w))
    #
    augimage = tf.concat([augR,augG,augB], axis =3)
    #
    # clip
    augimage = tf.clip_by_value(augimage,0.0,1.0)
    # scale image back into 0-255 range
    augimage = tf.multiply(augimage,255.0)
    # return augmented image
    return augimage

# ---------------------------------------------------------------- #
# color temperature/color balance augmentation functions
# ---------------------------------------------------------------- ##

def aug_color(image_rgb, a_transl, b_transl):
    #
    # Convert image to CIE L*a*b* color space
    # https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    #
    # a_transl = tf.expand_dims(tf.constant(a_transl,dtype=tf.float32), axis = 2)
    # b_transl = tf.expand_dims(tf.constant(b_transl,dtype=tf.float32), axis = 2)
    #
    # normalize the image between 0 and 1, convert to float
    image_ = tf.image.convert_image_dtype(image_rgb/255.0, tf.float32)
    #
    # convert image to LAB color space
    image_lab = rgb_to_lab(image_)
    #
    # split into the 3 lab channels
    Lchan, achan, bchan = tf.split(image_lab, 3, axis= 3)
    #
    # apply transformations in the a and b axes
    aug_a = achan+a_transl
    aug_b = bchan+b_transl
    #
    # convert back to rgb colorspace
    auglab_ = tf.squeeze(tf.stack([Lchan, aug_a, aug_b], axis=3))
    #auglab_ = image_lab
    augim_rgb = lab_to_rgb(auglab_)
    #
    #scale back to 0-255 range
    augimage = tf.multiply(augim_rgb, 255.0)
    #
    return augimage

### ---------------------------------------------------------------- ###
### --------------- noise augmentation functions ------------------- ### 
### ---------------------------------------------------------------- ###

def aug_noise(image_rgb, batchsize, Ra_sd, Rb_si, Ga_sd, Gb_si, Ba_sd, Bb_si, im_h, im_w):
    #
    # Based upon the noise model presented in Foi et al 2009
    # Noise is modeled by the addition of a poisson (signal-dependent noise) and gaussian distribution (independent noise)
    #
    # a_sd = tf.constant(Ra_sd,shape=(batchsize,1,1,1), dtype = tf.float32)
    # b_si = tf.constant(Rb_si,shape=(batchsize,1,1,1), dtype = tf.float32) 

    ## BAYER VARIABLE DEFINITION ##
    # define matrix that captures photosite bleeding effects
    #photobleed = tf.constant(np.array([[0.95,0.04,0.01],[0.07,0.89,0.04],[0.0,0.06,0.94]]), dtype=tf.float32, shape = (1,3,3,1))
    #
    # define bayer cfa architecture
    bayer_type='GBRG'
    # define the cfa/bayer pattern (sensor locations) for each color channel
    Rcfa, Gcfa, Bcfa = return_bayer(bayer_type, im_h, im_w, batchsize) 
    # define cfa interpolation kernels
    RandB_interp = 0.25*np.array([[1,2,1],[2,4,2],[1,2,1]])
    G_interp = 0.25*np.array([[0,1,0],[1,4,1],[0,1,0]])
    Rcfa_kernel = tf.constant(RandB_interp, dtype=tf.float32, shape = (3,3,1,1))
    Gcfa_kernel = tf.constant(G_interp, dtype=tf.float32, shape = (3,3,1,1))
    Bcfa_kernel = tf.constant(RandB_interp, dtype=tf.float32, shape = (3,3,1,1))
    #
    # normalize images
    image_rgb_ = tf.image.convert_image_dtype(image_rgb/255.0, tf.float32)
    #
    ## model photosite bleeding in image ##
    #image_prgb = tf.squeeze(tf.tensordot(image_rgb_, photobleed, axes=[[3],[2]]))
    #
    # split the image into its channels
    Rchan,Gchan,Bchan = tf.split(image_rgb_, 3, axis=3)
    #
    ## add in realistic sensor noise to each channel ##
    Rchan_ = add_channel_noise(Rchan, Ra_sd, Rb_si, batchsize, im_h, im_w)
    Gchan_ = add_channel_noise(Gchan, Ga_sd, Gb_si, batchsize, im_h, im_w)
    Bchan_ = add_channel_noise(Bchan, Ba_sd, Bb_si, batchsize, im_h, im_w)
    #
    ## add in effects from bilinear interpolation on bayer cfa ##
    Rchan__ = bilinear_interp_cfa(Rchan_, Rcfa, Rcfa_kernel, batchsize, im_h, im_w)
    Gchan__ = bilinear_interp_cfa(Gchan_, Gcfa, Gcfa_kernel, batchsize, im_h, im_w)
    Bchan__ = bilinear_interp_cfa(Bchan_, Bcfa, Bcfa_kernel, batchsize, im_h, im_w)
    #
    # compose the noisy image:
    augnoise = tf.concat([Rchan__,Gchan__,Bchan__],axis=3) 
    # scale image to 0-255
    augnoise = tf.multiply(augnoise, 255.0)
    #
    augimg = augnoise
    #pdb.set_trace()
    return augimg
    #

def add_channel_noise(chan, a_sd, b_si, batchsize, im_h, im_w):
    ##
    ## determine sensor noise at each pixel using non-clipped poisson-gauss model from FOI et al 
    ##
    if a_sd==0.0:
        chi=0
        sigdep = chan
    else:
        chi = 1.0/a_sd
        rate = tf.maximum(chi*chan,0)
        sigdep = tf.random_poisson(rate, shape=[])/chi
        #
    sigindep = tf.sqrt(b_si)*tf.random_normal(shape=(batchsize, im_h, im_w, 1), mean=0.0, stddev=1.0)
    # sum the two noise sources
    chan_noise = sigdep + sigindep
    #
    #sigdep = tf.sqrt(a_sd*chan)*tf.random_normal(shape=(batchsize,im_h, im_w, 1), mean=0.0, stddev=1.0)
    #sigindep = tf.sqrt(b_si)*tf.random_normal(shape=(batchsize,im_h, im_w, 1), mean=0.0, stddev=1.0)
    #chan_noise = chan + sigdep + sigindep
    #
    #chan_noise = chan + tf.sqrt(a_sd*chan + b_si)*tf.random_normal(shape=(batchsize,im_h, im_w, 1), mean=0.0, stddev=1.0)
    #
    # clip the noise between 0 and 1 (baking in 0 and 255 limits)
    clip_chan_noise = tf.clip_by_value(chan_noise, 0.0, 1.0)
    #
    return clip_chan_noise

def bilinear_interp_cfa(chan, cfa, cfa_kernel,batchsize, im_h, im_w):
    #
    # calculate pixel intensities based upon bayer CFA pattern
    #
    # location of pixel sensors for this color channel in the bayer array
    pix_mask = tf.equal(cfa,tf.constant(1)) 
    pix_is = chan
    pix_not = tf.zeros_like(chan)
    # get values of specific color channel sensors based upon the geometry/location of the cfa/bayer color sensors
    pix_on_cfa = tf.where(pix_mask, pix_is, pix_not)
    # use basic bilinear interpolation to solve for the noise that is in the non-pixel sensor locations
    interp_pixs = tf.nn.conv2d(pix_on_cfa, cfa_kernel, strides=[1, 1, 1, 1], padding='SAME')
    #
    #pdb.set_trace()
    return interp_pixs

def return_bayer(bayer_type, im_h, im_w, batchsize):
    #
    # generate the CFA arrays for R,G,B based upon the r pixel location:
    # 
    if bayer_type=='BGGR':
        # bggr
        Cr=np.array([[1,0],[0,0]])
        Cg=np.array([[0,1],[1,0]])
        Cb=np.array([[0,0],[0,1]])
        Rcfa= np.tile( Cr, (im_h/2,im_w/2))
        Gcfa= np.tile( Cg, (im_h/2,im_w/2))
        Bcfa= np.tile( Cb, (im_h/2,im_w/2))
        #
    if bayer_type=='GBRG':
        ## gbrg
        Cr2=np.array([[0,1],[0,0]])
        Cg2=np.array([[1,0],[0,1]])
        Cb2=np.array([[0,0],[1,0]])
        Rcfa= np.tile( Cr2, (im_h/2,im_w/2))
        Gcfa= np.tile( Cg2, (im_h/2,im_w/2))
        Bcfa= np.tile( Cb2, (im_h/2,im_w/2))
        #
    if bayer_type=='GRBG':
        ## grbg
        Cr3=np.array([[0,0],[1,0]])
        Cg3=np.array([[1,0],[0,1]])
        Cb3=np.array([[0,1],[0,0]])
        Rcfa= np.tile( Cr3, (im_h/2,im_w/2))
        Gcfa= np.tile( Cg3, (im_h/2,im_w/2))
        Bcfa= np.tile( Cb3, (im_h/2,im_w/2))
        #
    if bayer_type=='RGGB':
        ## rggb
        Cr4=np.array([[0,0],[0,1]])
        Cg4=np.array([[0,1],[1,0]])
        Cb4=np.array([[1,0],[0,0]])
        Rcfa= np.tile( Cr4, (im_h/2,im_w/2))
        Gcfa= np.tile( Cg4, (im_h/2,im_w/2))
        Bcfa= np.tile( Cb4, (im_h/2,im_w/2))
        #
    Rcfa= np.tile( Rcfa, (batchsize,1,1))
    Gcfa= np.tile( Gcfa, (batchsize,1,1))
    Bcfa= np.tile( Bcfa, (batchsize,1,1))
    #
    Rcfa = tf.constant(Rcfa, dtype=tf.int32, shape = (batchsize,im_h,im_w,1))
    Gcfa = tf.constant(Gcfa, dtype=tf.int32, shape = (batchsize,im_h,im_w,1))
    Bcfa = tf.constant(Bcfa, dtype=tf.int32, shape = (batchsize,im_h,im_w,1))
    #
    return Rcfa, Gcfa, Bcfa 

# ---------------------------------------------------------------- #
# Exposure augmentation functions
# ---------------------------------------------------------------- ##
def aug_exposure(image, delta_S, A, batchsize):
    #
    # Exposure
    #
    ## contrast variable
    #A = 0.85
    ## calculate the exposure change
    ##delta_S = tf.constant(delta_S, shape = (batchsize,1,1,1),dtype = tf.float32)
    #
    # normalize image between 0 and 1
    hin = tf.add(tf.divide(image,255.0),0.0001)
    # project image into exposure space
    S = tf.divide(tf.log(tf.subtract(tf.divide(255.0,hin),1.0)),-A)
    # translate image in exposure space
    Sprime = tf.add(S,delta_S);
    # project augmented image back into original image space
    Iprime = tf.divide(255.0,tf.add(1.0,tf.exp(tf.multiply(-A,Sprime))))
    # clip
    Iprime = tf.clip_by_value(Iprime,0.0,1.0)
    # scale augmented image to 0->255 range
    hout = tf.multiply(Iprime,255.0)
    #
    return hout

def aug_exposure_simple(image, delta_S, Amin, Amax, batchsize):
    #
    # scaling and shifting the luminance channel
    #
    # normalize the image between 0 and 1, convert to float
    image_ = tf.image.convert_image_dtype(image_rgb/255.0, tf.float32)
    # convert image to LAB color space
    image_lab = rgb_to_lab(image_)
    # split into the 3 lab channels
    Lchan, achan, bchan = tf.split(image_lab, 3, axis= 3)
    #
    # shift the L channel
    Laug = Lchan + delta_S
    #
    # scale the L channel
    Laugnorm = (Laug - tf.reduce_min(Laug,axis = [1,2,3]))/(tf.reduce_max(Laug,axis = [1,2,3])-tf.reduce_min(Laug,axis = [1,2,3]))
    Laug2 = Amax*Laugnorm + Amin
    # convert back to rgb colorspace
    auglab_ = tf.squeeze(tf.stack([Laug, aug_a, aug_b], axis=3))
    auglab_ = tf.clip_by_value(auglab_,0.0,1.0)
    augim_rgb = lab_to_rgb(auglab_)
    #scale back to 0-255 range
    augimage = tf.multiply(augim_rgb, 255.0)
    #
    return augimage

# ---------------------------------------------------------------- #
# Blur augmentation functions
# ---------------------------------------------------------------- ##
#
def aug_blur(img_inp, window_l, sig_arr, batchsize):
    #
    # Blur
    # Iprime = cv2.GaussianBlur(image,(hsize,hsize),sigma)
    #

    # normalize image to 0-1 range and convert to float
    image_norm = tf.image.convert_image_dtype(img_inp/255.0, tf.float32)
    # get batches
    batch_list = tf.split(image_norm,batchsize, axis=0)
    batch_counter=0
    conv_batch_list=[]
    for bimg in batch_list:
        # split channels
        Rchan,Gchan,Bchan = tf.split(bimg, 3, axis= 3)
        # get the window and sigma
        wl = tf.squeeze(window_l[batch_counter])
        sig = tf.squeeze(sig_arr[batch_counter])
        # get the kernel - [fh,fw,chan_in, chan_out]
        fgauss = gaussiankern2D(wl, sig)
        # conv
        R_conv = tf.nn.conv2d(Rchan, fgauss, strides=[ 1, 1, 1, 1], padding='SAME')
        G_conv = tf.nn.conv2d(Gchan, fgauss, strides=[ 1, 1, 1, 1], padding='SAME')
        B_conv = tf.nn.conv2d(Bchan, fgauss, strides=[ 1, 1, 1, 1], padding='SAME')
        bimg_conv= tf.stack([R_conv, G_conv, B_conv], axis=3)
        conv_batch_list.append(tf.squeeze(bimg_conv))
        batch_counter+=1
        #
    img_conv= tf.stack(conv_batch_list,0)
    augimage = tf.squeeze(img_conv)
    # clip
    augimage = tf.clip_by_value(augimage,0.0,1.0)
    # scale image back into 0-255 range
    augimage = tf.multiply(augimage,255.0)
    # return augmented image
    return augimage

def disckern2D(disc_radius,wl):
    # generate a disc kernel with radius disc_radius
    disc_kernel = tf.ones((2*wl, 2*wl, 1, 1))
    x, y = tf.meshgrid(2*disc_radius, 2*disc_radius)
    #determine where to make the elements zero
    disc_kernel[x*x + y*y >= r*r] = 0
    return disc_kernel_


def boxkern2D(wl):
    boxkern = (1/wl**2)*tf.ones((wl,wl))
    return tf.expand_dims(tf.expand_dims(boxkern, axis = 2), axis = 3)

def gaussiankern2D(wl, sig):
    """
    creates gaussian kernel with side length window_length and a sigma of sigma
    """
    # [filter_height, filter_width, in_channels, out_channels]
    # initialize filter
    wx = tf.range(-wl/2 + 1., wl/2 + 1.) #np.arange(-wl // 2 + 1., wl // 2 + 1.)
    xx, yy = tf.meshgrid(wx, wx)#np.meshgrid(wx, wx)
    tkernel = tf.exp(-(xx**2 + yy**2) / (2. * sig**2))#np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    tkernel_ = tkernel / tf.reduce_sum(tkernel)#tkernel / np.sum(tkernel)
    expkernel_ = tf.expand_dims(tf.expand_dims(tkernel_,axis=2), axis =3)
    #
    return expkernel_

### ---------------------------------------------------------------------------------------------------------------------------



