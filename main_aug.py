import os
import scipy.misc
import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import camGAN
from utils import pp
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "number of epochs; corresponds to number of augmentations to perform on the dataset (i.e., epoch =2 means the dataset will be augmented twice")
flags.DEFINE_integer("batch_size", 2, "The size of batch images; must be a multiple of n and >1")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
#
flags.DEFINE_string("Img_dataset","generator_images","The name (full path) of dataset to augment")
flags.DEFINE_integer("Img_height",512, "The size of the output images to produce [64]")
flags.DEFINE_integer("Img_width", 1024, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_boolean("chromab_flag", True, "flag that specifies whether to perform Chromatic aberration augmentation")
flags.DEFINE_boolean("blur_flag", True, "flag that specifies whether to perform Blur augmentation")
flags.DEFINE_boolean("exposure_flag", True, "flag that specifies whether to perform Exposure augmentation")
flags.DEFINE_boolean("noise_flag", True, "flag that specifies whether to perform noise augmentation")
flags.DEFINE_boolean("color_flag", True, "flag that specifies whether to perform color shift augmentation")
flags.DEFINE_boolean("save_aug_params_flag", False, "flag that specifies whether to save aug. parameters for each image")
#
flags.DEFINE_string("input_fname_pattern", "*.png", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("results_dir", "results", "Directory name to save the augmented images [results]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)
  ##
  if FLAGS.Img_width is None:
    FLAGS.Img_width = FLAGS.Img_height
    ##
  if not os.path.exists(FLAGS.results_dir):
    os.makedirs(FLAGS.results_dir)
    ##
  run_config = tf.ConfigProto()
  ## allocate only as much GPU memory based on runtime allocations
  run_config.gpu_options.allow_growth=True
  ##
  with tf.Session(config=run_config) as sess:
    autoauggan = camGAN(
      sess,
      Img_width=FLAGS.Img_width,
      Img_height=FLAGS.Img_height,
      batch_size=FLAGS.batch_size,
      c_dim=FLAGS.c_dim,
      Img_dataset_name = FLAGS.Img_dataset,
      chromab_flag = FLAGS.chromab_flag,
      blur_flag = FLAGS.blur_flag,
      exposure_flag = FLAGS.exposure_flag,
      noise_flag = FLAGS.noise_flag,
      color_flag = FLAGS.color_flag,
      save_aug_params_flag = FLAGS.save_aug_params_flag,
      input_fname_pattern=FLAGS.input_fname_pattern,
      results_dir = FLAGS.results_dir)

    #if FLAGS.is_train:
    if True:
      autoauggan.augment_batches(FLAGS)
    else:
      if not autoauggan.load(FLAGS.checkpoint_dir):
        raise Exception("[!] Train a model first, then run test mode")
      #wgan.test(FLAGS)

if __name__ == '__main__':
  tf.app.run()
