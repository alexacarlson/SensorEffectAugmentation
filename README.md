##
## SENSOR EFFECT AUGMENTATION PIPELINE 
##

This repository contains the tensorflow implementation of the Sensor Effect Augmentation Pipeline described in *Modeling Camera Effects to Improve Visual Learning from Synthetic Data* (https://arxiv.org/abs/1803.07721).

### Setting up the Sensor Effect Pipeline
This pipeline uses docker/containers, so it can be run on any computer. 
For a good tutorial on docker see https://docs.docker.com/get-started/
The docker image used by this model needs to be built before being used, which is done by running the following 
command in the Sensor_Augmentation_Pipeline folder:

docker build -t tf-sensor-augment augment-docker-image/

where 'tf-sensor-augment' is the tag for the docker image and is what is used to start a container.
The docker image is specified in augment-docker-image/Dockerfile.


### Running the Pipeline/Augmenting Images

To run the pipeline, use the command

```$. run_main_aug.sh```

in the command line within the SensorEffectAugmentation folder.
You can customize the type of augmentations, the dataset to augmented, etc by modifying the ```run_main_aug.sh```, which is described in more detail below.

The Sensor Effect Image Augmentation Pipeline is comprised of the following files:

```run_main_aug.sh```

This is a bash file that initializes a docker container and runs the image augmentation pipeline.

The first volume mapping ```-v `pwd`/ImgData``` (line 2)
argument is the full path to the directory that holds the images you would like to augment. The default location is set to ```ImgData``` in the SensorEffectAugmentation directory.

The second volume mapping ```-v `pwd`/AugmentedImgData``` (line 3) is the path to the directory where the augmented images will be saved. The default location is set to ```AugmentedImgData``` in the SensorEffectAugmentation directory.

The other volume mappings are setting up the file system in the docker container. 

The command ```python /root/main.py``` (line 11) runs the image augmentation pipeline. 

The inputs to the ```main.py``` function are:

```--epoch``` the number of epochs; corresponds to number of augmentations to perform on the dataset i.e., epoch =2 means the dataset will be augmented twice

```--batch_size``` The size of batch images; must be a multiple of n and >1

```--c_dim``` Dimension of image color channel (note that any channel >3 will be discarded)

```--Img_dataset``` The name full path of dataset to augment

```--Img_height``` The size of the output images to produce (note that all images will be resized to the specified Img_height x Img_width)

```--Img_width``` The size of the output images to produce. If None, same value as output_height 

```--chromab_flag``` flag that specifies whether to perform Chromatic aberration augmentation

```--blur_flag``` flag that specifies whether to perform Blur augmentation

```--exposure_flag``` flag that specifies whether to perform Exposure augmentation

```--noise_flag``` flag that specifies whether to perform noise augmentation

```--color_flag``` flag that specifies whether to perform color shift augmentation

```--save_aug_params_flag``` flag that specifies whether to save aug. parameters for each image

```--input_fname_pattern``` Glob pattern of filename of input images 

```--results_dir``` Directory name to save the augmented images

```main_aug.py```
	This is a master function that handles input flags and initializing the augmentation. It is called by ```run_main_aug.sh```.

```model_aug.py```
	This is a python module that defines the image augmentation class and builds the tensorflow graph. It is called by ```main_aug.py```.

```augmentationfunctions_tf.py```
	This is a python module that contains all of the sensor augmentation functions for blur, chromatic aberration, exposure changes, sensor noise, and color shifts.
	Please see our arxiv paper (https://arxiv.org/abs/1803.07721) for more information on these functions.

```geometric_transformation_module.py```
	This is a python module that implements affine warping; is used for augmenting images with chromatic aberration via warping the R and B color channels relative to the G channel.
	It is called by augmentationfunctions_tf.py

```pix2pix_lab2RGBconv.py```
	This is a python module that implements color conversion between RGB and LAB color spaces. It is called by augmentationfunctions_tf.py

Examples of GTA images augmented with the above sensor effects are located in ```Sensor_Augmentation_Pipeline/augFIGURES``` folder.




