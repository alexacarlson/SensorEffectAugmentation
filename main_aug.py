import os
import scipy.misc
import numpy as np
from model import camGAN
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Augment a dataset')
parser.add_argument('-n', type=int, default=1, nargs='?', help='sets the number of augmentations to perform on the dataset i.e., setting n to 2 means the dataset will be augmented twice')
parser.add_argument('-b', '--batch_size', type=int, default=64, nargs='?', help='size of batches; must be a multiple of n and >1')
parser.add_argument('-c', '--channels', type=int, default=3, nargs='?', help='dimension of image color channel (note that any channel >3 will be discarded')
parser.add_argument('-i', '--input', type=str, help='path to the dataset to augment')
parser.add_argument('-o', '--output', type=str, default='results', nargs='?', help='path where the augmented dataset will be saved')
parser.add_argument('--pattern', type=str, default="*.png", nargs='?', help='glob pattern of filename of input images')
parser.add_argument('--image_height', type=int, default=512, nargs='?', help='size of the output images to produce (note that all images will be resized to the specified image_height x image_width)')
parser.add_argument('--image_width', type=int, default=1024, nargs='?', help='size of the output images to produce. If None, same value as output_height')
parser.add_argument('--chromatic_aberration', type=bool, default=False, nargs='?', help='perform chromatic aberration augmentation')
parser.add_argument('--blur', type=bool, default=False, nargs='?', help='perform blur augmentation')
parser.add_argument('--exposure', type=bool, default=False, nargs='?', help='perform exposure augmentation')
parser.add_argument('--noise', type=bool, default=False, nargs='?', help='perform noise augmentation')
parser.add_argument('--colour_shift', type=bool, default=False, nargs='?', help='perform colour shift augmentation')
parser.add_argument('--save_params', type=bool, default=False, nargs='?', help='save augmentation parameters for each image')
args = parser.parse_args()

def main(_):
  print(args)
  ##
  if args.image_width is None:
    args.image_width = args.image_height
    ##
  if not os.path.exists(args.output):
    os.makedirs(args.output)
    ##
  run_config = tf.ConfigProto()
  ## allocate only as much GPU memory based on runtime allocations
  run_config.gpu_options.allow_growth=True
  ##
  with tf.Session(config=run_config) as sess:
    autoauggan = camGAN(
      sess,
      image_width=args.image_width,
      image_height=args.image_height,
      batch_size=args.batch_size,
      channels=args.channels,
      input = args.input,
      chromatic_aberration = args.chromatic_aberration,
      blur = args.blur,
      exposure = args.exposure,
      noise = args.noise,
      colour_shift = args.colour_shift,
      save_params = args.save_params,
      pattern=args.pattern,
      output = args.output)

    #if args.is_train:
    if True:
      autoauggan.augment_batches(args)
    else:
      if not autoauggan.load(args.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")
      #wgan.test(args)

if __name__ == '__main__':
  tf.app.run()
